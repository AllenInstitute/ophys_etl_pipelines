from typing import Optional
import numpy as np


def estimate_std_from_interquartile_range(
        data: np.ndarray,
        axis: Optional[int] = None) -> float:
    """
    Take a numpy.array and estimate the standard deviation of
    its contents as

    (75th_percentile - 25th_percentile)/1.34896

    Parameters
    ----------
    data: np.ndarray

    axis: Optional[int]
       default = None

    Returns
    -------
    float
    """
    q25, q75 = np.quantile(data, (0.25, 0.75), axis=axis)
    return (q75-q25)/1.34896
