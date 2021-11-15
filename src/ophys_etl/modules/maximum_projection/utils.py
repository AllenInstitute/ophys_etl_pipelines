import numpy as np
import scipy.ndimage as scipy_ndimage


def filter_chunk_of_frames(
        chunk_of_frames: np.ndarray,
        kernel_size: int) -> np.ndarray:
    """
    Apply a median filter to a subset of frames

    Parameters
    ----------
    chunk_of_frames: np.ndarray
        array of shape (ntime, nrows, ncols)

    kernel_size: int
        side length of square kernel to use for median filter

    Returns
    -------
    filtered_frames: np.ndarray
        (ntime, nrows, ncols) with each frame passed through
        the median filter
    """

    filtered_frames = np.zeros(chunk_of_frames.shape,
                               dtype=float)

    for i_frame in range(filtered_frames.shape[0]):
        filtered_frames[i_frame, :, :] = scipy_ndimage.median_filter(
                                             chunk_of_frames[i_frame, :, :],
                                             size=kernel_size,
                                             mode='reflect')
    return filtered_frames
