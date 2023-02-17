from typing import List
import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage.filters import median_filter


def nanmedian_filter(input_arr: np.ndarray, filter_length: int) -> np.array:
    """ 1D median filtering with np.nanmedian
    Parameters
    ----------
    input_arr: np.ndarray
        1d array of signal
    filter_length: int
        Length of the median filter to compute a rolling baseline

    Return
    ------
    filtered_trace
    """

    half_length = int(filter_length/2)
    # Create 'reflect' traces at the extrema
    temp_trace = np.concatenate(
        (
            np.flip(input_arr[:half_length]),
            input_arr,
            np.flip(input_arr[-half_length:]))
        )
    filtered_trace = np.zeros_like(input_arr)
    for i in range(len(input_arr)):
        filtered_trace[i] = np.nanmedian(temp_trace[i:i+filter_length])
    if np.isnan(filtered_trace).any():
        filtered_trace = _fill_nan(filtered_trace)
    return filtered_trace


def _fill_nan(input_arr: np.ndarray) -> np.ndarray:
    """Fill nan values in an array with interpolation
    Parameters
    ----------
    input_arr: np.ndarray
        1d array of signal containing nan values

    Returns
    -------
    np.ndarray
        array with filled nan values
    """
    nan_mask = np.isnan(input_arr)
    nan_indices = np.where(nan_mask)[0]
    no_nan_indices = np.where(~nan_mask)[0]
    interpolated_values = np.interp(
        nan_indices, no_nan_indices, input_arr[no_nan_indices])
    input_arr[nan_mask] = interpolated_values
    return input_arr


def robust_std(input_arr: np.ndarray) -> float:
    """Compute the median absolute deviation assuming normally
    distributed data. This is a robust statistic.

    Parameters
    ----------
    input_arr: np.ndarray
        A numeric, 1d numpy array
    Returns
    -------
    float:
        A robust estimation of standard deviation.
    Notes
    -----
    If `input_arr` is an empty array or contains any NaNs, will return NaN.
    """
    mad = np.median(np.abs(input_arr - np.median(input_arr)))
    return 1.4826*mad


def noise_std(input_arr: np.ndarray, filter_length: int = 31) -> float:
    """Compute a robust estimation of the standard deviation of the
    noise in a signal `input_arr`. The noise is left after subtracting
    a rolling median filter value from the signal. Outliers are removed
    in 2 stages to make the estimation robust.

    Parameters
    ----------
    input_arr: np.ndarray
        1d array of signal (perhaps with noise)
    filter_length: int (default=31)
        Length of the median filter to compute a rolling baseline,
        which is subtracted from the signal `input_arr`. Must be an odd number.

    Returns
    -------
    float:
        A robust estimation of the standard deviation of the noise.
        If any valurs of `input_arr` are NaN, returns NaN.
    """
    if any(np.isnan(input_arr)):
        return np.NaN
    noise = input_arr - median_filter(input_arr, filter_length)
    # first pass removing positive outlier peaks
    # TODO: Confirm with scientific team that this is really what they want
    # (method is fragile if possibly have 0 as min)
    filtered_noise_0 = noise[noise < (1.5 * np.abs(noise.min()))]
    rstd = robust_std(filtered_noise_0)
    # second pass removing remaining pos and neg peak outliers
    filtered_noise_1 = filtered_noise_0[abs(filtered_noise_0) < (2.5 * rstd)]
    return robust_std(filtered_noise_1)


def extract_traces(movie_frames: np.ndarray, rois: List[coo_matrix],
                   normalize_by_roi_size: bool = True,
                   block_size: int = 1000) -> np.ndarray:
    """Extract per ROI fluorescence traces from a movie.

    NOTE: Because the AllenSDK extract traces function does a
    mask size normalization and because it can't deal with weighted
    or coo_matrix format ROIs, this is an alternative implementation
    in order to make a fair comparison between traces generated using
    binarized vs weighted ROI masks.

    NOTE: The AllenSDK implementation should take precedence over this one.
    If an extract traces implementation is required in the full production
    pipeline.

    See: allensdk.brain_observatory.roi_masks.py #409-468

    Parameters
    ----------
    movie_frames : np.ndarray
        2P microscopy movie data: (frames x rows x cols)
    rois : List[coo_matrix]
        A list of ROIs in coo_matrix format.
    normalize_by_roi_size : bool, optional
        Whether to normalize traces by number of ROI elements, by default True.
        This is the behavior of the AllenSDK `calculate_traces` function.
    block_size : int, optional.
        The number of frames at a time to apply trace extraction to. Necessary
        for reasonable performance with hdf5 datasets.

    Returns
    -------
    np.ndarray
        ROI traces: (num_rois x frames)
    """
    num_frames = movie_frames.shape[0]

    traces = np.zeros((len(rois), len(movie_frames)))

    for frame_indx in range(0, num_frames, block_size):
        time_slice = slice(frame_indx, frame_indx + block_size)
        movie_slice = movie_frames[time_slice]

        for indx, roi in enumerate(rois):
            raw_trace = np.dot(movie_slice[:, roi.row, roi.col], roi.data)

            if normalize_by_roi_size:
                # Normalize by number of nonzero elements in ROI
                traces[indx, time_slice] = raw_trace / len(roi.data)
            else:
                traces[indx, time_slice] = raw_trace

    return traces
