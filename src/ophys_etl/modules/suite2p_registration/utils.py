import numpy as np
from typing import Tuple, List
from scipy.ndimage.filters import median_filter

import h5py
from suite2p.registration.rigid import shift_frame


def shift_movie_chunk(frames: List[int],
                      xoffs: int,
                      yoffs: int,
                      clipped_xoffs: int,
                      clipped_yoffs: int,
                      clipped_indices: List[int],
                      movie_path: str) -> np.ndarray:
    """
    """
    with h5py.File(movie_path, "r") as f:
        movie = f['data'][frames, :, :]
        data = []

        for frame in frames:
            if frame not in clipped_indices:
                data.append(shift_frame(
                    movie[frame - frames[0], :, :],
                    xoffs[frame],
                    yoffs[frame]))
            else:
                data.append(shift_frame(
                    movie[frame - frames[0], :, :],
                    clipped_xoffs[frame],
                    clipped_yoffs[frame]))

    return np.stack(data, axis=0)


def projection_process(data: np.ndarray,
                       projection: str = "max") -> np.ndarray:
    """

    Parameters
    ----------
    data: np.ndarray
        nframes x nrows x ncols, uint16
    projection: str
        "max" or "avg"

    Returns
    -------
    proj: np.ndarray
        nrows x ncols, uint8

    """
    if projection == "max":
        proj = np.max(data, axis=0)
    elif projection == "avg":
        proj = np.mean(data, axis=0)
    else:
        raise ValueError("projection can be \"max\" or \"avg\" not "
                         f"{projection}")
    proj = np.uint8(proj * 255.0 / proj.max())
    return proj


def identify_and_clip_outliers(data: np.ndarray,
                               med_filter_size: int,
                               thresh: int) -> Tuple[np.ndarray, List]:
    """given data, identify the indices of outliers
    based on median filter detrending, and a threshold

    Parameters
    ----------
    data: np.ndarray
        1D array of samples
    med_filter_size: int
        the number of samples for 'size' in
        scipy.ndimage.filters.median_filter
    thresh: int
        multipled by the noise estimate to establish a threshold, above
        which, samples will be marked as outliers.

    Returns
    -------
    data: np.ndarry
        1D array of samples, clipped to threshold around median-filtered data
    indices: np.ndarray
        the indices where clipping took place

    """
    data_filtered = median_filter(data, med_filter_size, mode='nearest')
    detrended = data - data_filtered
    indices = np.argwhere(np.abs(detrended) > thresh).flatten()
    data[indices] = np.clip(data[indices],
                            data_filtered[indices] - thresh,
                            data_filtered[indices] + thresh)
    return data, indices
