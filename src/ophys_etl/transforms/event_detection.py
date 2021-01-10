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
from scipy.optimize import NonlinearConstraint, minimize
from functools import partial
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
    legacy_bracket = argschema.fields.Bool(
        required=False,
        default=False,
        description=("whether to run the legacy recursive bracketing method "
                     "for regularization search, vs. the new scipy-based "
                     "search."))

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
        data['halflife'] = calculate_halflife(data['decay_time'])
        return data


def fast_lzero(penalty: float, dat: np.ndarray, gamma: float,
               constraint: bool) -> np.ndarray:
    """runs fast spike inference and returns an array like the input trace
    with data substituted by event magnitudes

    Parameters
    ----------
    penalty: float
        tuning parameter lambda > 0
    dat: np.ndarray
        fluorescence data
    gamma: float
        a scalar value for the AR(1) decay parameter; 0 < gam <= 1
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
    out[ev['spikes']] = ev['pos_spike_mag']
    return out


def calculate_gamma(halflife: float, sample_rate: float) -> float:
    """calculate gamma from halflife and sample rate.

    Parameters
    ----------
    halflife: float
        halflife [seconds]
    sample_rate: float
        sample rate [Hz]

    Returns
    -------
    gamma: float
        attenuation factor from exponential decay over 1 time sample

    """
    lam = np.log(2) / (halflife * sample_rate)
    gamma = np.exp(-lam)
    return gamma


def calculate_halflife(decay_time: float) -> float:
    """conversion from decay_time to halflife

    Parameters
    ----------
    decay_time: float
        also known as mean lifetime in [seconds]

    Returns
    -------
    float
        halflife in [seconds]
    """
    return np.log(2) * decay_time


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
        passed as size to scipy.ndimage.filters.median_filter

    Returns
    -------
    float
        estimate of the standard deviation. np.NaN if as NaN's are in
        incoming data

    """
    x = x - median_filter(x, filt_length, mode='nearest')
    x = x[x < 1.5 * np.abs(x.min())]
    rstd = median_abs_deviation(x, scale='normal')
    x = x[abs(x) < 2.5 * rstd]
    return median_abs_deviation(x, scale='normal')


def fast_lzero_regularization_search_scipy(
        trace: np.ndarray, noise_estimate: float,
        gamma: float, penalty_guess: float = 0.1) -> Tuple[np.ndarray, float]:
    """finds events through a regularization search, subject to the
    constraints of having n_events > 0 (establishing the upper bound for
    regularization penalty) and the smallest event magnitude near the noise
    estimate.

    Parameters
    ----------
    trace: np.ndarray
        the 1D trace data
    noise_estimate: float
        a std-dev estimate of the noise, used to establish the smallest
        desired event magnitude.
    gamma: float
        a scalar value for the AR(1) decay parameter; 0 < gam <= 1
    penalty_guess: float
        initial guess for the regularization penalty

    Returns
    -------
    events: np.ndarray
        array of event magnitudes, the same size as trace
    penalty: float
        the optimized regularization parameter

    Notes
    -----
    - The original code also had a parameter that would multiply noise
    estimate. This parameter (event_min_size) was always set to 1.
    If desired later, this can be added as an additional argument here.
    - This search benefits from using established scipy.optimize methods,
    making it readable, extensible, and thus maintainable. There is an
    inefficiency in that `fast_lzero_partial` is executed twice per iteration,
    (is this accurate? this depends how the underlying Trust region method is
    searching the space) and a final time to retrieve the events. Since the
    underlying optimization from FastLZeroSpikeInference is indeed fast
    (a few seconds per trace), this seems like a reasonable tradeoff.
    If one wanted to use this scipy-based method and avoid the repeat calls,
    one could implement an lru_cache on the `fast_lzero` method, but, some
    additional work to make the inputs and outputs hashable would be required.

    """
    fast_lzero_partial = partial(fast_lzero, dat=trace,
                                 gamma=gamma, constraint=True)

    def n_event_call(penalty):
        events = fast_lzero_partial(penalty)
        n_events = np.count_nonzero(events)
        return n_events

    def min_mag_call(penalty):
        events = fast_lzero_partial(penalty)
        n_events = np.count_nonzero(events)
        if n_events == 0:
            # we want to drive regularization smaller
            min_mag = noise_estimate * 1000
        else:
            min_mag = events[events > 0].min()
        metric = np.power(min_mag - noise_estimate, 2)
        return metric

    # constrain number of events to be > 0 (i.e. >= 0.5)
    nonlinear_constraint = NonlinearConstraint(n_event_call, 0.5, np.inf)
    optimize_result = minimize(min_mag_call,
                               [penalty_guess],
                               method='trust-constr',
                               constraints=nonlinear_constraint)

    penalty = optimize_result.x[0]
    events = fast_lzero_partial(penalty)

    return events, penalty


def bracket(dff, noise, s1, step, step_min, gamma, bisect=False):
    # NOTE: in original code, 1.0 was parametrized, but it was unused
    # and L0_constrain was set to True
    min_size = 1.0 * noise
    L0_constrain = True

    penalty = s1 + step
    if penalty < step:
        penalty = step
        s1 += step
    tmp = fast_lzero(penalty, dff, gamma, penalty)
    print(penalty, len(tmp[tmp > 0]), np.min(tmp[tmp > 0]))

    if len(tmp[tmp > 0]) == 0 and bisect is True:
        # if there are not events, move penalty down
        # by half a step
        return bracket(dff, noise, s1 - 5*step, step, step_min, gamma)

    if step == step_min:
        if np.min(tmp[tmp > 0]) > min_size and bisect is True:
            return bracket(dff, noise, s1 - 5*step, step, step_min, gamma)
        else:
            while (len(tmp[tmp > 0]) > 0 and
                   np.min(tmp[tmp > 0]) < min_size):
                lasttmp = tmp[:]
                penalty += step
                tmp = fast_lzero(penalty, dff, gamma, L0_constrain)
            if len(tmp[tmp > 0]) == 0:
                return (lasttmp, penalty - step)
            else:
                return (tmp, penalty)

    if len(tmp[tmp > 0]) == 0 and bisect is False:
        # if there are not events, move penalty down
        # and shrink step size
        return bracket(
                dff, noise, s1 + .5*step - step/10,
                step/10, step_min, gamma, bisect=True)

    if len(tmp[tmp > 0]) > 0 and np.min(tmp[tmp > 0]) < min_size:
        return bracket(dff, noise, penalty,
                       step, step_min, gamma)

    condition = (len(tmp[tmp > 0]) > 0 and
                 np.min(tmp[tmp > 0]) > min_size and
                 step > step_min and bisect is False)
    if condition:
        # there are events, we're not at minimum step
        # move penalty down, reduce step size
        return bracket(
                dff, noise, s1 + .5 * step - step/10,
                step/10, step_min, gamma, bisect=True)

    condition = (len(tmp[tmp > 0]) > 0 and
                 np.min(tmp[tmp > 0]) > min_size and
                 step > step_min and bisect is True)
    if condition:
        # there are events, we're not at minimum step
        # move penalty down, set bisect True
        return bracket(
                dff, noise, s1 - 5 * step,
                step, step_min, gamma)


def estimate_noise_detrend(traces: np.ndarray,
                           filter_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trace: estimates noise and median filters with a noise-based
    clipping threshold of the median.

    Parameters
    ----------
    traces: np.ndarray
        ntrace x nframes, float
    filter_size: int
        length of median filter for detrending, passed to
        scipy.ndimage.filters.median_filter as size.

    Returns
    -------
    traces: np.ndarray
        ntrace x nframes, detrended traces
    sigmas: np.ndarray
        size ntrace, float, estimate of trace noise

    Notes
    -----
    original code used median_filter mode of 'constant' which pads for
    the median calculation with zero. In many cases, this led to a small
    detected event at the end of the trace. Changing to 'nearest' mode
    eliminates this behavior.

    """
    sigmas = np.empty(shape=traces.shape[0])
    for i, trace in enumerate(traces):
        sigmas[i] = trace_noise_estimate(trace)
        trace_median = median_filter(trace, filter_size, mode='nearest')
        # NOTE the next line clips the median trace from above
        # there is no stated motivation in the original code.
        trace_median = np.minimum(trace_median, 2.5 * sigmas[i])
        trace -= trace_median

    return traces, sigmas


def get_events(traces, noise_estimates, gamma, legacy_bracket=False):
    if not legacy_bracket:
        results = [fast_lzero_regularization_search_scipy(
                       trace=trace, noise_estimate=sigma, gamma=gamma)
                   for trace, sigma in zip(traces, noise_estimates)]
    else:
        results = [bracket(trace, sigma, 0, 0.1, 0.0001, gamma)
                   for trace, sigma in zip(traces, noise_estimates)]

    events = [result[0] for result in results]
    events = np.array(events)
    lambdas = [result[1] for result in results]
    return np.array(events), lambdas


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
        short_median_filter = 101
        dff, noise_stds = estimate_noise_detrend(
                upsampled_dff, short_median_filter)
        gamma = calculate_gamma(self.args['halflife'],
                                upsampled_rate)
        upsampled_events, lambdas = get_events(
                dff, noise_stds, gamma,
                legacy_bracket=self.args['legacy_bracket'])

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
            f.create_dataset("noise_stds", data=noise_stds)
            f.create_dataset("lambdas", data=lambdas)
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
