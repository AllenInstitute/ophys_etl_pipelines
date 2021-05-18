import numpy as np
from typing import Union
from scipy.ndimage import uniform_filter1d, gaussian_filter1d


def temporal_filter1d(video: np.ndarray,
                      size: Union[int, float],
                      filter_type: str = "uniform") -> np.ndarray:
    """apply simple averaging per pixel trace

    Parameters
    ----------
    video: np.ndarray
        the input video nframes x nrows x ncols
    size: int or float
        passed to 'uniform_filter1d' as 'size' (int) or to 'gaussian_filter1d'
        as 'sigma' (float)
    filter_type: str
        'uniform' or 'gaussian'

    Returns
    -------
    output: np.ndarray
        the filtered video, same size and shape as the input video

    """
    filters = {
            "uniform": uniform_filter1d,
            "gaussian": gaussian_filter1d}
    if filter_type not in filters:
        raise ValueError("'filter_type' must be one of "
                         f"{list(filters.keys())}, "
                         f"but {filter_type} was provided.")
    if filter_type == "uniform":
        size = int(size)
    output = filters[filter_type](video, size, axis=0, mode="nearest")
    return output
