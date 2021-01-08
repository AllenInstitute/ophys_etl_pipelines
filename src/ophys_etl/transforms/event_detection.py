import argschema
import h5py
import os
import numpy as np
import json
import marshmallow as mm
from scipy.ndimage.filters import median_filter
from scipy.signal import resample_poly
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
        return data


def medfilt(x, s):
    return median_filter(x, s, mode='constant')


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
                 infer_lambda=False,
                 compute_dff=False):

        self.use_bisection = use_bisection
        self.median_filter_1 = median_filter_1
        self.median_filter_2 = median_filter_2
        self.L0_constrain = L0_constrain
        self.compute_dff = compute_dff
        self.infer_lambda = infer_lambda

        try:
            self.metadata = dataset.get_metadata()
            self.corrected_fluorescence_traces = \
                dataset.get_corrected_fluorescence_traces()[1]
            self.num_cells = self.corrected_fluorescence_traces.shape[0]
            self._dff_traces = None
            self._noise_stds = None
            self._num_small_baseline_frames = None
        except:
            self.metadata = {
                    'ophys_experiment_id': 999}
            self._dff_traces = dataset
            self.num_cells = self._dff_traces[0]
            num_small_baseline_frames = []
            noise_stds = []

            for dff in self._dff_traces:
                if self.compute_dff:
                    sigma_f = self.noise_std(dff)

                    # long timescale median filter for baseline subtraction
                    tf = medfilt(dff, self.median_filter_1)
                    dff -= tf
                    dff /= np.maximum(tf, sigma_f)

                    num_small_baseline_frames.append(np.sum(tf <= sigma_f))

                sigma_dff = self.noise_std(dff)
                noise_stds.append(sigma_dff)

                # short timescale detrending
                tf = medfilt(dff, self.median_filter_2)
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

    def l0(self, dff, gamma, l, constraint):
        ev = fast.estimate_spikes(dff, gamma, l, constraint,
                                  estimate_calcium=True)
        out = np.zeros(ev['dat'].shape)
        out[ev['spikes']-1] = ev['pos_spike_mag']
        return out

    @property
    def dff_traces(self):
        self.min_detected_event_sizes = [[] for n in range(self._dff_traces.shape[0])]
        return self._dff_traces, self._noise_stds, self._num_small_baseline_frames

    @property
    def gamma(self):
        if self._gamma is None:
            self._gamma = np.exp(-np.log(2) * 1000 /
                                 (self.halflife * self.sample_rate_hz))
        return self._gamma

    def noise_std(self, x, filt_length=31):
        if any(np.isnan(x)):
            return np.NaN
        x = x - medfilt(x, filt_length)
        # first pass removing big pos peak outliers
        x = x[x < 1.5 * np.abs(x.min())]
        rstd = self.robust_std(x)
        # second pass removing remaining pos and neg peak outliers
        x = x[abs(x) < 2.5*rstd]
        return self.robust_std(x)

    def robust_std(self, x):
        '''
        Robust estimate of std
        '''
        MAD = np.median(np.abs(x - np.median(x)))
        return 1.4826*MAD

    def get_events(self, event_min_size=None, use_bisection=None):
        if event_min_size is not None:
            self.event_min_size = event_min_size

        if use_bisection is not None:
            self.use_bisection = use_bisection

        self.print('Calculating events in progress', flush=True)

        events = []
        # parallelization written by PL
        results = Parallel(n_jobs=40, prefer="threads")(delayed(self.get_event_trace)((n, dff)) for n, dff in enumerate(self.dff_traces[0]))
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
                l = np.NaN
            else:
                tmp = dff[:]
                if self.infer_lambda:
                    # empirical lambda(noise_std) fit
                    fit = [3.63986409e+00, 1.81463579e-03, 3.56562092e-05]
                    l = (fit[0] *
                         self._noise_stds[n]**2 +
                         fit[1] * self._noise_stds[n] + fit[2])
                    tmp = self.l0(
                            self.dff_traces[1][n],
                            self.gamma, l, self.L0_constrain)
                elif self.use_bisection:
                    (tmp, l) = self.bisection(
                            tmp, self.dff_traces[1][n], self.event_min_size)
                else:
                    (tmp, l) = self.bracket(
                            tmp, self.dff_traces[1][n], 0,
                            .1, .0001, self.event_min_size)
            return (tmp, l)
        except Exception as e:
            print(e)
            tmp = np.NaN*np.zeros(dff.shape)
            l = np.NaN
            return (tmp, l)

    def bisection(self, dff, n, event_min_size,
                  left=0., right=5., max_its=100, eps=.0001):

        # find right endpoint with no events
        tmp_right = self.l0(dff, self.gamma, right, self.L0_constrain)
        nz_right = (tmp_right > 0)

        # bisection for lambda minimizing num events < min size
        it = 0
        while it <= max_its:

            it += 1
            if (right - left) < eps:
                break

            mid = left + (right - left) / 2.

            tmp_left = self.l0(dff, self.gamma, left, self.L0_constrain)
            nz_left = (tmp_left > 0)
            num_small_events_left = np.sum(
                    tmp_left[nz_left] < n*event_min_size)

            if num_small_events_left == 0:
                break
            else:
                tmp_mid = self.l0(dff, self.gamma, mid, self.L0_constrain)
                tmp_right = self.l0(dff, self.gamma, right, self.L0_constrain)

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
        l = s1 + step
        if l < step:
            l = step
            s1 += step
        tmp = self.l0(dff, self.gamma, l, self.L0_constrain)

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
                    l += step
                    tmp = self.l0(dff, self.gamma, l, self.L0_constrain)
                if len(tmp[tmp > 0]) == 0:
                    return (lasttmp, l-step)
                else:
                    return (tmp, l)

        if len(tmp[tmp > 0]) == 0 and bisect is False:
            return self.bracket(
                    dff, n, s1 + .5*step - step/10,
                    step/10, step_min, event_min_size, True)

        if len(tmp[tmp > 0]) > 0 and np.min(tmp[tmp > 0]) < n * event_min_size:
            return self.bracket(dff, n, l, step, step_min, event_min_size)

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


class EventDetection(argschema.ArgSchemaParser):
    default_schema = EventDetectionInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        self.logger.setLevel(self.args.pop('log_level'))
        ophys_etl_commit_sha = os.environ.get("OPHYS_ETL_COMMIT_SHA",
                                              "local build")
        self.logger.info(f"OPHYS_ETL_COMMIT_SHA: {ophys_etl_commit_sha}")

        with h5py.File(self.args['ophysdfftracefile'], 'r') as f:
            inds = np.argwhere(
                    np.isin(
                        f['roi_names'][()],
                        self.args['valid_roi_ids'])).flatten()
            cids = f['roi_names'][inds]
            dff = f['data'][inds]

        fs = self.args['movie_frame_rate_hz']
        uf = np.round(30.9 / fs)  # upsampling factor

        event_dict = {}
        halflife_ms = 1000 * np.log(2) * self.args['decay_time']

        if uf == 1:  # If resampling is unnecessary (e.g. Scientifica sessions)
            l0a = L0_analysis(
                    dff,
                    sample_rate_hz=fs,
                    halflife_ms=halflife_ms)
            events = l0a.get_events()
            for n, event in enumerate(events):
                event_mag = event[event > 0]  # event magnitudes
                event_idx = np.where(event > 0)
                event_dict[cids[n]] = {
                        'mag': event_mag, 'idx': event_idx,
                        'event_trace': event}  # linking events to cids
            np.savez_compressed(
                    self.args['output_event_file'],
                    dff=dff,
                    events=events,
                    noise_stds=l0a._noise_stds,
                    lambdas=l0a.lambdas,
                    event_dict=event_dict,
                    upsampling_factor=uf)
        else:  # resampling is necessary (e.g. mesoscope sessions)
            dff30Hz = resample_poly(dff, uf, 1, axis=1)
            l0a = L0_analysis(
                    dff30Hz,
                    sample_rate_hz=fs*uf,
                    halflife_ms=halflife_ms)
            events30Hz = l0a.get_events()
            events = np.zeros_like(dff)  # same size as the original dff
            for n, event30Hz in enumerate(events30Hz):
                event_mag = event30Hz[event30Hz > 0]  # upsampled
                event_idx = np.where(event30Hz > 0)[0]  # upsampled
                for i, ev in enumerate(event_idx):
                    # downsample event magnitudes
                    events[n, np.int(ev/uf)] += event_mag[i]
                event_dict[cids[n]] = {
                        'mag': event_mag, 'idx': event_idx,
                        'event_trace': events[n, :]}  # linking events to cids
            np.savez_compressed(
                    self.args['output_event_file'],
                    dff=dff,
                    events=events,
                    noise_stds=l0a._noise_stds,
                    lambdas=l0a.lambdas,
                    event_dict=event_dict,
                    upsampling_factor=uf)


if __name__ == "__main__":  # pragma: no cover
    ed = EventDetection()
    ed.run()
