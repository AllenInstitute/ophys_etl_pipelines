import argschema
import h5py
import os
import numpy as np
import json
import multiprocessing
import warnings
import marshmallow as mm
from typing import Tuple
from scipy.ndimage.filters import median_filter
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
        description="characteristic decay time [seconds]")
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
    noise_median_filter = argschema.fields.Float(
        required=False,
        default=1.0,
        description=("median filter length used to detrend data "
                     "during noise estimation [seconds]. Typically "
                     "shorter than 'trace_median_filter'."))
    trace_median_filter = argschema.fields.Float(
        required=False,
        default=3.2,
        description=("median filter length used to detrend data "
                     "before passing to FastLZero. [seconds]. Typically "
                     "longer than 'noise_median_filter'."))
    noise_multiplier = argschema.fields.Float(
        required=False,
        description=("manual specification of noise multiplier. If not "
                     "provided, will be defaulted by `get_noise_multiplier` "
                     "post_load below."))

    @mm.post_load
    def check_dff_h5_keys(self, data, **kwargs):
        with h5py.File(data['ophysdfftracefile'], 'r') as f:
            if 'roi_names' not in list(f.keys()):
                raise EventDetectionException(
                        f"DFF trace file {data['ophysdfftracefile']} "
                        "does not have the key 'roi_names', which indicates "
                        "it has come from an old version of creating "
                        "DFF traces < April, 2019. Consider recreating the "
                        "DFF file with a current version of that module.")
        return data

    @mm.post_load
    def get_noise_multiplier(self, data, **kwargs):
        if 'noise_multiplier' in data:
            return data
        # NOTE 11Hz value found empirically to behave well compared
        # to 31Hz data sub-sampled to 11Hz.
        val_11hz = 2.6
        val_31hz = 2.0
        if np.round(data['movie_frame_rate_hz'] / 11.0) == 1:
            data['noise_multiplier'] = val_11hz
        elif np.round(data['movie_frame_rate_hz'] / 31.0) == 1:
            data['noise_multiplier'] = val_31hz
        else:
            raise EventDetectionException(
                "You did not specify 'noise_multiplier'. In that case, the "
                "multiplier is selected by 'movie_frame_rate_hz' but only "
                "freqeuncies around 11Hz and 31Hz have specified values: "
                f"({val_11hz}, {val_31hz}) respectively. You specified a "
                f"'movie_frame_rate' of {data['movie_frame_rate_hz']}")
        return data

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
            lookup = decay_lookup[data['full_genotype']]
            if 'decay_time' in data:
                warnings.warn("You specified a decay_time of "
                              f"{data['decay_time']} but that is being "
                              "overridden by a lookup by genotype to give "
                              f"a decay time of {lookup}.")
            data['decay_time'] = lookup
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
        a scalar value for the AR(1) decay parameter; 0 < gamma <= 1
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


def trace_noise_estimate(x: np.ndarray, filt_length: int) -> float:
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
    x = x[np.abs(x) < 2.5 * rstd]
    return median_abs_deviation(x, scale='normal')


def count_and_minmag(events: np.ndarray) -> Tuple[int, float]:
    """from an array of events count the number of events
    and get the minimum magnitude

    Parameters
    ----------
    events: np.ndarray
        size nframes, zeros where no event, otherwise magnitude

    Returns
    -------
    n_events: int
        number of nonzero entries
    min_event_mag: float
        magnitude of smallest non-zero event

    """
    n_events = len(events[events > 0])
    if n_events == 0:
        min_event_mag = 0.0
    else:
        min_event_mag = np.min(events[events > 0])
    return n_events, min_event_mag


def fast_lzero_regularization_search_bracket(
        trace: np.ndarray, min_size: float, gamma: float,
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
    min_size: float
        a target for the minimum amplitude magnitude. This is typically a
        multiple of an estimate of the noise.
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
    L0_constrain = True

    # these args are immutable through the recursion
    recursive_partial = partial(fast_lzero_regularization_search_bracket,
                                trace, min_size, gamma,
                                step_min=step_min)

    # evaluate fast_lzero at one point
    penalty = base_penalty + penalty_step
    if penalty < penalty_step:
        penalty = penalty_step
        base_penalty += penalty_step
    flz_eval = fast_lzero(penalty, trace, gamma, L0_constrain)
    n_events, min_event_mag = count_and_minmag(flz_eval)

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
            while n_events > 0 and min_event_mag < min_size:
                last_flz_eval = np.copy(flz_eval)
                penalty += penalty_step
                flz_eval = fast_lzero(penalty, trace, gamma, L0_constrain)
                n_events, min_event_mag = count_and_minmag(flz_eval)
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


def estimate_noise_detrend(traces: np.ndarray, noise_filter_size: int,
                           trace_filter_size: int
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trace: estimates noise and median filters with a noise-based
    clipping threshold of the median.

    Parameters
    ----------
    traces: np.ndarray
        ntrace x nframes, float
    noise_filter_size: int
        length of median filter for detrending during noise estimation,
        passed to scipy.ndimage.filters.median_filter as size.
    trace_filter_size: int
        length of median filter for detrending data,
        passed to scipy.ndimage.filters.median_filter as size.

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
        sigmas[i] = trace_noise_estimate(trace, noise_filter_size)
        trace_median = median_filter(trace, trace_filter_size, mode='nearest')
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
            roi_names = f['roi_names'][()].astype('int')
            valid_roi_indices = np.argwhere(
                    np.isin(
                        roi_names,
                        self.args['valid_roi_ids'])).flatten()
            dff = f['data'][valid_roi_indices]
        valid_roi_names = roi_names[valid_roi_indices]

        empty_warning = None
        if valid_roi_names.size == 0:
            events = np.empty_like(dff)
            lambdas = np.empty(0)
            noise_stds = np.empty(0)
            empty_warning = ("No valid ROIs in "
                             f"{self.args['ophysdfftracefile']}. "
                             "datasets in output file will be empty.")
            self.logger.warn(empty_warning)
        else:
            # run FastLZeroSpikeInference
            noise_filter_samples = int(self.args['noise_median_filter'] *
                                       self.args['movie_frame_rate_hz'])
            trace_filter_samples = int(self.args['trace_median_filter'] *
                                       self.args['movie_frame_rate_hz'])
            dff, noise_stds = estimate_noise_detrend(
                    dff,
                    noise_filter_size=noise_filter_samples,
                    trace_filter_size=trace_filter_samples)
            gamma = calculate_gamma(self.args['halflife'],
                                    self.args['movie_frame_rate_hz'])
            events, lambdas = get_events(
                    traces=dff,
                    noise_estimates=noise_stds * self.args['noise_multiplier'],
                    gamma=gamma,
                    ncpu=self.args['n_parallel_workers'])

        with h5py.File(self.args['output_event_file'], "w") as f:
            f.create_dataset("events", data=events)
            f.create_dataset("roi_names", data=valid_roi_names)
            f.create_dataset("noise_stds", data=noise_stds)
            f.create_dataset("lambdas", data=lambdas)
            if empty_warning:
                f.create_dataset("warning", data=empty_warning)
        self.logger.info(f"wrote {self.args['output_event_file']}")


if __name__ == "__main__":  # pragma: no cover
    ed = EventDetection()
    ed.run()
