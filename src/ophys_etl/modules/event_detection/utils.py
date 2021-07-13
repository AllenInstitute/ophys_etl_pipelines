import numpy as np
from typing import Tuple
from scipy.ndimage.filters import median_filter
from scipy.stats import median_abs_deviation


class EventDetectionException(Exception):
    pass


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
