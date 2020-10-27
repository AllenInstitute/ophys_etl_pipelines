from typing import List, Tuple
from functools import partial

import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage.filters import median_filter


# Partial for simplifying repeat median filter calls
medfilt = partial(median_filter, mode='constant')


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


def robust_std(x: np.ndarray) -> float:
    """Compute the median absolute deviation assuming normally
    distributed data. This is a robust statistic.

    Parameters
    ----------
    x: np.ndarray
        A numeric, 1d numpy array
    Returns
    -------
    float:
        A robust estimation of standard deviation.
    Notes
    -----
    If `x` is an empty array or contains any NaNs, will return NaN.
    """
    mad = np.median(np.abs(x - np.median(x)))
    return 1.4826*mad


def noise_std(x: np.ndarray, filter_length: int = 31) -> float:
    """Compute a robust estimation of the standard deviation of the
    noise in a signal `x`. The noise is left after subtracting
    a rolling median filter value from the signal. Outliers are removed
    in 2 stages to make the estimation robust.

    Parameters
    ----------
    x: np.ndarray
        1d array of signal (perhaps with noise)
    filter_length: int (default=31)
        Length of the median filter to compute a rolling baseline,
        which is subtracted from the signal `x`. Must be an odd number.

    Returns
    -------
    float:
        A robust estimation of the standard deviation of the noise.
        If any valurs of `x` are NaN, returns NaN.
    """
    if any(np.isnan(x)):
        return np.NaN
    noise = x - medfilt(x, filter_length)
    # first pass removing positive outlier peaks
    # TODO: Confirm with scientific team that this is really what they want
    # (method is fragile if possibly have 0 as min)
    filtered_noise_0 = noise[noise < (1.5 * np.abs(noise.min()))]
    rstd = robust_std(filtered_noise_0)
    # second pass removing remaining pos and neg peak outliers
    filtered_noise_1 = filtered_noise_0[abs(filtered_noise_0) < (2.5 * rstd)]
    return robust_std(filtered_noise_1)


def compute_dff_trace(corrected_fluorescence_trace: np.ndarray,
                      long_filter_length: int,
                      short_filter_length: int
                      ) -> Tuple[np.ndarray, float, int]:
    """
    Compute the "delta F over F" from the fluorescence trace.
    Uses configurable length median filters to compute baseline for
    baseline-subtraction and short timescale detrending.
    Returns the artifact-corrected and detrended dF/F, along with
    additional metadata for QA: the estimated standard deviation of
    the noise ("sigma_dff") and the number of frames where the
    computed baseline was less than the standard deviation of the noise.

    Parameters
    ----------
    corrected_fluorescence_trace: np.array
        1d numpy array of the corrected fluorescence trace
    long_filter_length: int
        Length (in number of elements) of the long median filter used
        to compute a rolling baseline. Must be an odd number.
    short_filter_length: int (default=31)
        Length (in number of elements) for a short median filter used
        for short timescale detrending.
    Returns
    -------
    np.ndarray:
        The "dff" (delta_fluorescence/fluorescence) trace, 1d np.array
    float:
        The estimated standard deviation of the noise in the dff trace
    int:
        Number of frames where the baseline (long timescape median
        filter) was less than or equal to the estimated noise of the
        `corrected_fluorescence_trace`.
    """
    sigma_f = noise_std(corrected_fluorescence_trace)

    # Long timescale median filter for baseline subtraction
    baseline = medfilt(corrected_fluorescence_trace, long_filter_length)
    dff = ((corrected_fluorescence_trace - baseline)
           / np.maximum(baseline, sigma_f))
    num_small_baseline_frames = np.sum(baseline <= sigma_f)

    sigma_dff = noise_std(dff)

    # Short timescale detrending
    filtered_dff = medfilt(dff, short_filter_length)
    # Constrain to 2.5x the estimated noise of dff
    filtered_dff = np.minimum(filtered_dff, 2.5*sigma_dff)
    detrended_dff = dff - filtered_dff

    return detrended_dff, sigma_dff, num_small_baseline_frames
