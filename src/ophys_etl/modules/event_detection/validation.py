import numpy as np
from typing import Union, List


def make_event(n_samples: int, index: int, magnitude: float,
               decay_time: float, rate: float) -> np.ndarray:
    """create a time series of an event (spike)

    Parameters
    ----------
    n_samples: int
        the number of samples
    index: int
        the index marking the start of the event
    magnitude: float
        the peak magnitude of the spike
    decay_time: float
        the characteristic decay time of the event [seconds]
    rate: float
        sampling rate in [Hz]

    Returns
    -------
    z: np.ndarray
        the time series with the event of size n_samples

    """
    timestamps = np.arange(n_samples) / rate
    t0 = timestamps[index]
    z = np.zeros(n_samples)
    z[index:] = magnitude * np.exp(-(timestamps[index:] - t0) / decay_time)
    return z


def sum_events(n_samples: int, timestamps: Union[List[int], np.ndarray],
               magnitudes: Union[List[float], np.ndarray], decay_time: float,
               rate: float) -> np.ndarray:
    """given a list of timestamps and magnitudes, create and sum events into
    a single timeseries

    Parameters
    ----------
    n_samples: int
        the number of samples in the timeseries
    timestamps: List[int], or np.ndarray
        the indices marking the start of the event
    magnitudes: List[float], or np.ndarray
        the peak magnitudes of the events, same length as timestamps
    decay_time: float
        the characteristic decay time of the event [seconds]
    rate: float
        sampling rate in [Hz]

    Returns
    -------
    data: np.ndarray
        the time series with the events, of size n_samples

    """
    data = np.zeros(n_samples)
    for ts, mag in zip(timestamps, magnitudes):
        data += make_event(n_samples, ts, mag, decay_time, rate)
    return data
