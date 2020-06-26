from typing import List

import numpy as np
from scipy.sparse import coo_matrix


def extract_traces(movie_frames: np.ndarray, rois: List[coo_matrix],
                   normalize_by_roi_size: bool = True,
                   block_size: int = 1000) -> np.ndarray:
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
