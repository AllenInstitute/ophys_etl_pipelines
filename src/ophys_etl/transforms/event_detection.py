import argschema
import h5py
import os
import numpy as np
import json
import multiprocessing
import marshmallow as mm
from typing import Tuple
from scipy.ndimage.filters import median_filter
from scipy.signal import resample_poly
from scipy.stats import median_abs_deviation
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
    n_parallel_workers = argschema.fields.Int(
        required=False,
        default=1,
        description=("number of parallel workers. If set to -1 "
                     "is set to multiprocessing.cpu_count()."))

    @mm.post_load
    def cpu_to_the_max(self, data, **kwargs):
        if data['n_parallel_workers'] == -1:
            data['n_parallel_workers'] = multiprocessing.cpu_count()
        return data

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
        estimate of the standard deviation.

    """
    x = x - median_filter(x, filt_length, mode='nearest')
    x = x[x < 1.5 * np.abs(x.min())]
    rstd = median_abs_deviation(x, scale='normal')
    x = x[abs(x) < 2.5 * rstd]
    return median_abs_deviation(x, scale='normal')


def fast_lzero_regularization_search_bracket(
        trace: np.ndarray, noise_estimate: np.ndarray, gamma: float,
        base_penalty: float = 0, penalty_step: float = 0.1,
        step_min: float = 0.0001,
        bisect=False) -> Tuple[np.ndarray, np.ndarray]:
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
    base_penalty: float
        initial guess for the regularization penalty
    penalty_step: float
        penalty step size
    step_min: float
        minimum possible step size
    bisect: bool
        when True, invokes a large step

    Returns
    -------
    events: np.ndarray
        array of event magnitudes, the same size as trace
    penalty: float
        the optimized regularization parameter

    Notes
    -----
    - This is the original Allen 'bracket' routine refactored to make it
    readable. The logic has not been changed.
    - In original, a multiplicative factor for the noise estimate was
    parametrized, but never used. Hard-coded here, for now, as the original
    default of 1.0.
    - In original, L0_contstrain (bool) was parametrized to tell FastLZero to
    perform LZero regularization, which was always True. Hard-coded
    here to True.

    """
    min_size = 1.0 * noise_estimate
    L0_constrain = True

    # these args are immutable through the recursion
    recursive_partial = partial(fast_lzero_regularization_search_bracket,
                                trace, noise_estimate, gamma,
                                step_min=step_min)

    # evaluate fast_lzero at one point
    penalty = base_penalty + penalty_step
    if penalty < penalty_step:
        penalty = penalty_step
        base_penalty += penalty_step
    flz_eval = fast_lzero(penalty, trace, gamma, L0_constrain)
    n_events = len(flz_eval[flz_eval > 0])
    min_event_mag = np.min(flz_eval[flz_eval > 0])

    if n_events == 0 and bisect is True:
        base_penalty -= 5 * penalty_step
        return recursive_partial(
                base_penalty=base_penalty,
                penalty_step=penalty_step)

    if penalty_step == step_min:
        if min_event_mag > min_size and bisect is True:
            base_penalty -= 5 * penalty_step
            return recursive_partial(
                    base_penalty=base_penalty,
                    penalty_step=penalty_step)
        else:
            while (n_events > 0 and min_event_mag < min_size):
                last_flz_eval = np.copy(flz_eval)
                penalty += penalty_step
                flz_eval = fast_lzero(penalty, trace, gamma, L0_constrain)
                n_events = len(flz_eval[flz_eval > 0])
                min_event_mag = np.min(flz_eval[flz_eval > 0])
            if n_events == 0:
                return (last_flz_eval, penalty - penalty_step)
            else:
                return (flz_eval, penalty)

    if n_events == 0 and bisect is False:
        base_penalty += 0.5 * penalty_step - penalty_step / 10
        penalty_step /= 10
        return recursive_partial(
                base_penalty=base_penalty,
                penalty_step=penalty_step,
                bisect=True)

    if n_events > 0 and min_event_mag < min_size:
        return recursive_partial(
                base_penalty=penalty,
                penalty_step=penalty_step)

    if n_events > 0 and min_event_mag > min_size and penalty_step > step_min:
        if bisect:
            base_penalty -= 5 * penalty_step
        else:
            base_penalty += 0.5 * penalty_step - penalty_step / 10
            penalty_step /= 10
        return recursive_partial(
                base_penalty=base_penalty,
                penalty_step=penalty_step,
                bisect=not bisect)


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


def get_events(traces: np.ndarray, noise_estimates: np.ndarray,
               gamma: float, ncpu: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """finds events through one of two methods for regularization search.
    parallelized by trace.

    Parameters
    ----------
    traces: np.ndarray
        ntrace x nframes, the trace data
    noise_estimates: np.ndarray
        length ntrace, estimates of standard deviations
    gamma: float
        decay-time dependent parameter, passed to FastLZero algorithm. The
        attenuation factor from exponential decay in one sampling step.
    ncpu: int
        number or workers for the multiprocessing pool

    Returns
    -------
    events: np.ndarray
        event magnitudes. same shape as traces. zero where no event detected.
    lambdas: np.ndarray
        length ntrace optimized regularization values.

    """
    args = [(trace, sigma, gamma)
            for trace, sigma in zip(traces, noise_estimates)]
    func = fast_lzero_regularization_search_bracket

    # pytest-cov does not play nice with Pool context manager so explicit
    # close/join to get code coverage report in subprocess calls
    pool = multiprocessing.Pool(ncpu)
    results = pool.starmap(func, args)
    pool.close()
    pool.join()

    events, lambdas = zip(*results)
    events = np.array(events)
    lambdas = np.array(lambdas)
    return events, lambdas


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
        to generalize FastLZero to other target rates.

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
        upsampled_dff, noise_stds = estimate_noise_detrend(
                upsampled_dff, short_median_filter)
        gamma = calculate_gamma(self.args['halflife'],
                                upsampled_rate)
        upsampled_events, lambdas = get_events(
                traces=upsampled_dff,
                noise_estimates=noise_stds,
                gamma=gamma,
                ncpu=self.args['n_parallel_workers'])

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
