import argschema
import h5py
import os
import numpy as np
import json
import marshmallow as mm
from typing import Tuple
from scipy.ndimage.filters import median_filter
from scipy.signal import resample_poly
from scipy.stats import median_abs_deviation
from joblib import Parallel, delayed
from FastLZeroSpikeInference import fast
from ophys_etl.resources import event_decay_lookup_dict as decay_lookup
from ophys_etl.schemas.fields import H5InputFile


class EventDetectionException(Exception):
    pass


class EventDetectionInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.Str(required=False, default="INFO")
    movie_frame_rate_hz = argschema.fields.Float(
        required=True,
        description=("frame rate of the ophys video / trace. "
                     "used to upsample slower rates to 31Hz."))
    full_genotype = argschema.fields.Str(
        required=False,
        description=("full genotype of the specimen. Used to look "
                     "up characteristic decay time from "
                     "ophys_etl.resources.event_decay_time_lookup"))
    decay_time = argschema.fields.Float(
        required=False,
        description=("characteristic decay time [seconds]"))
    ophysdfftracefile = H5InputFile(
        required=True,
        description=("h5 file containing keys `data` (nROI x nframe) and "
                     "`roi_names` (length = nROI) output by the df/F "
                     "computation."))
    valid_roi_ids = argschema.fields.List(
        argschema.fields.Int,
        required=True,
        cli_as_single_argument=True,
        description=("list of ROI ids that are valid, for which "
                     "event detection will be performed."))
    output_event_file = argschema.fields.OutputFile(
        required=True,
        description="location for output of event detection.")

    @mm.post_load
    def get_decay_time(self, data, **kwargs):
        if ('full_genotype' not in data) & ('decay_time' not in data):
            raise EventDetectionException(
                    "Must provide either `decay_time` or `full_genotype` "
                    "for decay time lookup. "
                    "Available lookup values are "
                    f"\n{json.dumps(decay_lookup, indent=2)}")
        if 'full_genotype' in data:
            if data['full_genotype'] not in decay_lookup:
                raise EventDetectionException(
                        f"{data['full_genotype']} not available. "
                        "Available lookup values are "
                        f"\n{json.dumps(decay_lookup, indent=2)}")
            data['decay_time'] = decay_lookup[data['full_genotype']]
        data['halflife_ms'] = calculate_halflife(data['decay_time'])
        return data


def fast_lzero(dat: np.ndarray, gamma: float,
               penalty: float, constraint: bool) -> np.ndarray:
    """runs fast spike inference and returns an array like the input trace
    with data substituted by event magnitudes, shifted 1 index earlier

    Parameters
    ----------
    dat: np.ndarray
        fluorescence data
    gamma: float
        a scalar value for the AR(1) decay parameter; 0 < gam <= 1
    penalty: float
        tuning parameter lambda > 0
    constraint: bool
        constrained (true) or unconstrained (false) optimization

    Returns
    -------
    out: np.ndarray
        event magnitude array, same shape as dat

    Notes
    -----
    https://github.com/jewellsean/FastLZeroSpikeInference/blob/cdfaade68ceb6aa15ec5003c460de4e0575f1d5f/python/FastLZeroSpikeInference/fast.py#L30  # noqa: E501

    """
    ev = fast.estimate_spikes(dat, gamma, penalty,
                              constraint, estimate_calcium=True)
    out = np.zeros(ev['dat'].shape)
    out[ev['spikes'] - 1] = ev['pos_spike_mag']
    return out


def calculate_gamma(halflife: float, sample_rate: float) -> float:
    """calculate gamma from halflife and sample rate
    """
    return np.exp(-np.log(2) * 1000 / (halflife * sample_rate))


def calculate_halflife(decay_time: float) -> float:
    return 1000 * np.log(2) * decay_time


def mad_std_estimate(x):
    """estimates the standard deviation by scipy.stats.median_abs_deviation
    """
    return 1.4826 * median_abs_deviation(x)


def trace_noise_estimate(x: np.ndarray,
                         filt_length: int = 31) -> float:
    """estimates noise of a signal by detrending with a median filter,
    removing positive spikes, eliminating outliers, and, using the
    median absolute deviation estimator of standard deviation.

    Parameters
    ----------
    x: np.ndarray
        1-D array of values
    filt_length: int
        passed as size to scipy.ndimage.filters.mediam_filter

    Returns
    -------
    float
        estimate of the standard deviation. np.NaN if as NaN's are in
        incoming data

    """
    if any(np.isnan(x)):
        return np.NaN
    x = x - median_filter(x, filt_length, mode='constant')
    x = x[x < 1.5 * np.abs(x.min())]
    rstd = mad_std_estimate(x)
    x = x[abs(x) < 2.5 * rstd]
    return mad_std_estimate(x)


class L0_analysis:
    """
    Class for performing L0 event detection using an automatically determined
    lambda value. lambda is chosen by finding smallest lambda such that the
    size of the smallest detected event is greater or equal to
    event_min_size*robust std of noise. If such a lambda cannot be found, it
    uses the largest lambda that returns some non-zero values.

    Parameters
    ----------
    dataset : a dataset object (returned from get_ophys_experiment_data)
              or ophys_experiment_id or raw data
    event_min_size : smallest allowable event in units
                     of noise std [default: 1.0]
    median_filter_1 : the length of the window for long time scale median
                      filter detrending to estimate dff from
                      corrected_fluorescence_traces [default: 5401]
    median_filter_2 : the length of the window for short time scale
                      median filter detrending [default: 101]
    halflife_ms : half-life of the indicator in ms,
                  used to override lookup [default: None]
    sample_rate_hz : sampling rate of data in Hz

    Attributes
    ----------
    noise_stds : estimates of the std of the noise for each trace
    lambdas : chosen lambda for each trace
    gamma : the gamma decay constant calculated from the half-life
    dff_traces : detrended df/f traces

    Examples
    --------
    >>> l0a = L0_analysis(dataset)
    >>> events = l0a.get_events()

    """
    def __init__(self,
                 dataset,
                 event_min_size=1.,
                 median_filter_1=5401,
                 median_filter_2=101,
                 halflife_ms=None,
                 sample_rate_hz=30,
                 L0_constrain=True,
                 use_bisection=False,
                 infer_lambda=False):

        self.use_bisection = use_bisection
        self.median_filter_1 = median_filter_1
        self.median_filter_2 = median_filter_2
        self.L0_constrain = L0_constrain
        self.infer_lambda = infer_lambda

        self.metadata = {
                'ophys_experiment_id': 999}
        self._dff_traces = dataset
        self.num_cells = self._dff_traces[0]
        num_small_baseline_frames = []
        noise_stds = []

        for dff in self._dff_traces:
            sigma_dff = trace_noise_estimate(dff)
            noise_stds.append(sigma_dff)

            # short timescale detrending
            tf = median_filter(dff, self.median_filter_2, mode='constant')
            tf = np.minimum(tf, 2.5*sigma_dff)
            dff -= tf

            self.print('.', end='', flush=True)

        self._noise_stds = noise_stds
        self._num_small_baseline_frames = num_small_baseline_frames

        self.sample_rate_hz = sample_rate_hz
        self.event_min_size = event_min_size
        self.halflife = halflife_ms
        self._fit_params = None
        self._gamma = None
        self.lambdas = []

    @property
    def dff_traces(self):
        self.min_detected_event_sizes = \
                [[] for n in range(self._dff_traces.shape[0])]
        return (self._dff_traces, self._noise_stds,
                self._num_small_baseline_frames)

    @property
    def gamma(self):
        return calculate_gamma(self.halflife, self.sample_rate_hz)

    def get_events(self, event_min_size=None, use_bisection=None):
        if event_min_size is not None:
            self.event_min_size = event_min_size

        if use_bisection is not None:
            self.use_bisection = use_bisection

        self.print('Calculating events in progress', flush=True)

        events = []
        # parallelization written by PL
        results = Parallel(
                n_jobs=40,
                prefer="threads")(delayed(
                    self.get_event_trace)(
                        (n, dff)) for n, dff in enumerate(self.dff_traces[0]))
        events = [result[0] for result in results]
        events = np.array(events)
        self.lambdas = [result[1] for result in results]
        # end of parallelization written by PL

        self.print('done!')
        return np.array(events)

    def get_event_trace(self, tpl, event_min_size=None,
                        use_bisection=None):
        # This function is for parallelisation of event detection
        n = tpl[0]
        dff = tpl[1]
        try:
            if any(np.isnan(dff)):
                tmp = np.NaN*np.zeros(dff.shape)
                penalty = np.NaN
            else:
                tmp = dff[:]
                if self.infer_lambda:
                    # empirical lambda(noise_std) fit
                    fit = [3.63986409e+00, 1.81463579e-03, 3.56562092e-05]
                    penalty = (fit[0] *
                               self._noise_stds[n]**2 +
                               fit[1] * self._noise_stds[n] + fit[2])
                    tmp = fast_lzero(
                            self.dff_traces[1][n],
                            self.gamma, penalty, self.L0_constrain)
                elif self.use_bisection:
                    (tmp, penalty) = self.bisection(
                            tmp, self.dff_traces[1][n], self.event_min_size)
                else:
                    (tmp, penalty) = self.bracket(
                            tmp, self.dff_traces[1][n], 0,
                            .1, .0001, self.event_min_size)
            return (tmp, penalty)
        except Exception as e:
            print(e)
            tmp = np.NaN*np.zeros(dff.shape)
            penalty = np.NaN
            return (tmp, penalty)

    def bisection(self, dff, n, event_min_size,
                  left=0., right=5., max_its=100, eps=.0001):

        # find right endpoint with no events
        tmp_right = fast_lzero(dff, self.gamma, right, self.L0_constrain)
        nz_right = (tmp_right > 0)

        # bisection for lambda minimizing num events < min size
        it = 0
        while it <= max_its:

            it += 1
            if (right - left) < eps:
                break

            mid = left + (right - left) / 2.

            tmp_left = fast_lzero(dff, self.gamma, left, self.L0_constrain)
            nz_left = (tmp_left > 0)
            num_small_events_left = np.sum(
                    tmp_left[nz_left] < n*event_min_size)

            if num_small_events_left == 0:
                break
            else:
                tmp_mid = fast_lzero(dff, self.gamma, mid, self.L0_constrain)
                tmp_right = fast_lzero(dff, self.gamma, right,
                                       self.L0_constrain)

                nz_mid = (tmp_mid > 0)
                nz_right = (tmp_right > 0)

                if np.sum(nz_mid) > 0:
                    num_small_events_mid = np.sum(
                            tmp_mid[nz_mid] < n*event_min_size)
                else:
                    num_small_events_mid = -np.infty

                if np.sum(nz_right) > 0:
                    num_small_events_right = np.sum(
                            tmp_right[nz_right] < n*event_min_size)
                else:
                    num_small_events_right = -np.infty

                print('lambda_left: ' + str(left))
                print('lambda_mid: ' + str(mid))
                print('lambda_right: ' + str(right))
                print('num events_left: ' + str(num_small_events_left))
                print('num events_mid: ' + str(num_small_events_mid))
                print('num events_right: ' + str(num_small_events_right))

                if np.sign(num_small_events_mid) == \
                        np.sign(num_small_events_left):
                    left = mid
                else:
                    right = mid

        return tmp_left, left

    def bracket(self, dff, n, s1, step,
                step_min, event_min_size, bisect=False):
        penalty = s1 + step
        if penalty < step:
            penalty = step
            s1 += step
        tmp = fast_lzero(dff, self.gamma, penalty, self.L0_constrain)

        if len(tmp[tmp > 0]) == 0 and bisect is True:
            return self.bracket(
                    dff, n, s1 - 5*step,
                    step, step_min, event_min_size)

        if step == step_min:
            if np.min(tmp[tmp > 0]) > n * event_min_size and bisect is True:
                return self.bracket(
                        dff, n, s1 - 5*step,
                        step, step_min, event_min_size)
            else:
                while (len(tmp[tmp > 0]) > 0 and
                       np.min(tmp[tmp > 0]) < n * event_min_size):
                    lasttmp = tmp[:]
                    penalty += step
                    tmp = fast_lzero(dff, self.gamma, penalty,
                                     self.L0_constrain)
                if len(tmp[tmp > 0]) == 0:
                    return (lasttmp, penalty - step)
                else:
                    return (tmp, penalty)

        if len(tmp[tmp > 0]) == 0 and bisect is False:
            return self.bracket(
                    dff, n, s1 + .5*step - step/10,
                    step/10, step_min, event_min_size, True)

        if len(tmp[tmp > 0]) > 0 and np.min(tmp[tmp > 0]) < n * event_min_size:
            return self.bracket(dff, n, penalty,
                                step, step_min, event_min_size)

        condition = (len(tmp[tmp > 0]) > 0 and
                     np.min(tmp[tmp > 0]) > n * event_min_size and
                     step > step_min and bisect is False)
        if condition:
            return self.bracket(
                    dff, n, s1 + .5 * step - step/10,
                    step/10, step_min, event_min_size, True)

        condition = (len(tmp[tmp > 0]) > 0 and
                     np.min(tmp[tmp > 0]) > n * event_min_size and
                     step > step_min and bisect is True)

        if condition:
            return self.bracket(
                    dff, n, s1 - 5 * step,
                    step, step_min, event_min_size)

    def print(self, *args, **kwargs):
        print(*args, **kwargs)


def trace_resample(traces: np.ndarray, input_rate: float,
                   target_rate: float = 30.9) -> Tuple[np.ndarray, float]:
    """determines a rounded upsample factor and upsamples the traces,
    if applicable.

    Parameters
    ----------
    traces: np.ndarray
        ncell x nframe float
    input_rate: float
        sampling rate of traces [Hz]
    target_rate: float
        desired resampled rate [Hz]

    Returns
    -------
    resampled_traces: np.ndarray
        ncell x n_upsampled_frames float
    upsample_factor: int
        factor used to upsample

    Raises
    ------
    NotImplementedError
        for any target rate != 30.9Hz. Evaluation of results is needed
        to generalize this function to other target rates.

    """
    if target_rate != 30.9:
        raise NotImplementedError("spike inference has only been validated "
                                  "at a target rate of 30.9Hz")
    upsample_factor = np.round(target_rate / input_rate)
    if upsample_factor == 1.0:
        resampled_traces = traces
    else:
        resampled_traces = resample_poly(traces, upsample_factor, 1, axis=1)
    return resampled_traces, upsample_factor


class EventDetection(argschema.ArgSchemaParser):
    default_schema = EventDetectionInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))
        ophys_etl_commit_sha = os.environ.get("OPHYS_ETL_COMMIT_SHA",
                                              "local build")
        self.logger.info(f"OPHYS_ETL_COMMIT_SHA: {ophys_etl_commit_sha}")

        # read in the input file
        with h5py.File(self.args['ophysdfftracefile'], 'r') as f:
            roi_names = f['roi_names'][()]
            valid_roi_indices = np.argwhere(
                    np.isin(
                        roi_names,
                        self.args['valid_roi_ids'])).flatten()
            dff = f['data'][valid_roi_indices]
        valid_roi_names = roi_names[valid_roi_indices]

        # upsample to 30.9 Hz
        upsampled_dff, upsample_factor = trace_resample(
                traces=dff,
                input_rate=self.args['movie_frame_rate_hz'],
                target_rate=30.9)
        upsampled_rate = self.args['movie_frame_rate_hz'] * upsample_factor
        self.logger.info("upsampled traces from "
                         f"{self.args['movie_frame_rate_hz']} to "
                         f"{upsampled_rate}")

        # run FastLZeroSpikeInference
        l0a = L0_analysis(
                upsampled_dff,
                sample_rate_hz=upsampled_rate,
                halflife_ms=self.args['halflife_ms'])
        upsampled_events = l0a.get_events()

        # downsample events back to original rate
        if upsample_factor == 1:
            downsampled_events = upsampled_events
        else:
            downsampled_events = np.zeros_like(dff)
            for n, event30Hz in enumerate(upsampled_events):
                event_mag = event30Hz[event30Hz > 0]  # upsampled
                event_idx = np.where(event30Hz > 0)[0]
                for i, ev in enumerate(event_idx):
                    downsampled_events[n, np.int(ev/upsample_factor)] += \
                            event_mag[i]

        with h5py.File(self.args['output_event_file'], "w") as f:
            f.create_dataset("events", data=downsampled_events)
            f.create_dataset("roi_names", data=valid_roi_names)
            f.create_dataset("noise_stds", data=l0a._noise_stds)
            f.create_dataset("lambdas", data=l0a.lambdas)
            f.create_dataset("upsampling_factor", data=upsample_factor)
        self.logger.info(f"wrote {self.args['output_event_file']}")

        # NOTE I think this should be provided by SDK API, if desired.
        # Saving dictionaries in hdf5 isn't great, and this seems
        # very redundant.
        # event_dict = {}
        # for n, event30Hz in enumerate(upsampled_events):
        #     event_mag = event30Hz[event30Hz > 0]
        #     event_idx = np.where(event30Hz > 0)[0]
        #     event_dict[valid_roi_names[n]] = {
        #             'mag': event_mag,
        #             'idx': event_idx,
        #             'event_trace': downsampled_events[n, :]}


if __name__ == "__main__":  # pragma: no cover
    ed = EventDetection()
    ed.run()
