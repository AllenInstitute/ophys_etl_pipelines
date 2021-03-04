from typing import List
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
