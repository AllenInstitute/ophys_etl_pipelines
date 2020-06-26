from typing import List

import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.transforms.roi_transforms import (roi_bounds, crop_roi_mask)


def extract_traces(movie_frames: np.ndarray, rois: List[coo_matrix],
                   normalize_by_roi_size: bool = True) -> np.ndarray:
    """Extract per ROI fluorescence traces from a movie.

    Parameters
    ----------
    movie_frames : np.ndarray
        2P microscopy movie data: (frames x rows x cols)
    rois : List[coo_matrix]
        A list of ROIs in coo_matrix format.
    normalize_by_roi_size : bool, optional
        Whether to normalize traces by number of ROI elements, by default True.
        This is the behavior of the AllenSDK `calculate_traces` function.

    Returns
    -------
    np.ndarray
        ROI traces: (num_rois x frames)
    """

    traces = np.zeros((len(rois), len(movie_frames)))

    for indx, roi in enumerate(rois):

        # Originally wanted to do: movie_frames[:, roi.row, roi.col]
        # But H5PY doesn't allow out of order indexing
        # To get around this, slice out the smallest rectangle around the
        # roi over time to get an 'roi cube'.
        min_row, max_row, min_col, max_col = roi_bounds(roi)
        roi_cube = movie_frames[:, min_row:max_row, min_col:max_col]
        # To get appropriate row and column selectors, the roi mask also needs
        # to be cropped in the exact same way. Now we can apply fancy
        # OOO indexing.
        roi = crop_roi_mask(roi)
        raw_trace = np.dot(roi_cube[:, roi.row, roi.col], roi.data)

        if normalize_by_roi_size:
            # Normalize by number of nonzero elements in ROI
            traces[indx, :] = raw_trace / len(roi.data)
        else:
            traces[indx, :] = raw_trace

    return traces
