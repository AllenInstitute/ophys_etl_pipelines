import pytest

import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.transforms import trace_transforms


@pytest.mark.parametrize("frames, rois, normalize_by_roi_size, expected", [
    # movie frames
    (np.array([np.array(range(0, 6)).reshape((2, 3)),
               np.array(range(7, 13)).reshape((2, 3)),
               np.array(range(13, 19)).reshape((2, 3))]),
     # rois
     [np.array([[0, 1, 0],
                [1, 0, 1]]),
      np.array([[1, 1, 0],
                [0, 0, 0]]),
      np.array([[0, 0, 1],
                [0, 0, 1]])],
     # normalize_by_roi_size
     False,
     # expected
     np.array([[9, 30, 48],
               [1, 15, 27],
               [7, 21, 33]])),

    (np.array([np.array(range(0, 6)).reshape((2, 3)),
               np.array(range(7, 13)).reshape((2, 3)),
               np.array(range(13, 19)).reshape((2, 3))]),
     [np.array([[0, 1, 0],
                [1, 0, 1]]),
      np.array([[1, 1, 0],
                [0, 0, 0]]),
      np.array([[0, 0, 1],
                [0, 0, 1]])],
     True,
     np.array([[3, 10, 16],
               [0.5, 7.5, 13.5],
               [3.5, 10.5, 16.5]])),

    (np.array([np.array(range(0, 6)).reshape((2, 3)),
               np.array(range(7, 13)).reshape((2, 3)),
               np.array(range(13, 19)).reshape((2, 3))]),
     [np.array([[0.0, 0.0, 0.0],
                [1.0, 2.0, 1.0]]),
      np.array([[0.0, 0.0, 0.0],
                [1.0, 3.0, 0.0]]),
      np.array([[0.0, 0.0, 0.5],
                [0.0, 0.5, 1.0]])],
     False,
     np.array([[16, 44, 68],
               [15, 43, 67],
               [8, 22.0, 34.0]])),
])
def test_extract_traces(frames, rois, normalize_by_roi_size, expected):
    rois = [coo_matrix(roi) for roi in rois]

    obtained = trace_transforms.extract_traces(frames, rois,
                                               normalize_by_roi_size)

    assert np.allclose(obtained, expected)
