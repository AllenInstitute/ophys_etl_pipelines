import pytest
import numpy as np
from scipy.sparse import coo_matrix
from scipy.ndimage.filters import median_filter
import ophys_etl.utils.traces as tx


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.zeros(10), 0.0),    # Zeros
        (np.ones(20), 0.0),     # All same, not zero
        (np.array([-1, -1, -1]), 0.0),   # Negatives
        (np.array([]), np.NaN),   # Empty
        (np.array([0, 0, np.NaN, 0.0]), np.NaN),     # Has NaN
        (np.array([1]), 0.0),    # Unit
        (np.array([-1, 2, 3]), 1.4826)    # Typical
    ]
)
def test_robust_std(x, expected):
    np.testing.assert_equal(expected, tx.robust_std(x))


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([0, 1, 2, 3, np.NaN]), np.NaN),
        (np.array([4, 10, 4, 5, 4]), 2.5)
    ])
def test_noise_std(x, expected, monkeypatch):
    monkeypatch.setattr(tx, "robust_std", lambda x: x.max())
    monkeypatch.setattr(tx, "median_filter", lambda x, y: x/2)
    np.testing.assert_equal(expected, tx.noise_std(x))


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

    obtained = tx.extract_traces(frames, rois, normalize_by_roi_size)

    assert np.allclose(obtained, expected)


@pytest.mark.parametrize("input_arr, size, expected", [
    # Equal to median_filter with reflect mode when no nans are present
    (
        np.arange(100),
        5,
        median_filter(np.arange(100), 5, mode='reflect'),
    ),
    # If block of nan values is as large filter size, fill in with previous
    # value
    (
        np.array([1, 2, 3, np.nan, np.nan, np.nan, 3, 2, 1]),
        3,
        np.array([1, 2, 2.5, 3, 3, 3, 2.5, 2, 1]),
    ),
    # If block of nan values is as large filter size, fill in with previous
    # value
    (
        np.array([np.nan, np.nan, np.nan, 5, 4, 3, 2, 1]),
        3,
        np.array([5, 5, 5, 4.5, 4, 3, 2, 1]),
    )
])
def test_nanmedian_filter(input_arr, size, expected):
    obtained = tx.nanmedian_filter(input_arr, size)
    np.testing.assert_equal(obtained, expected)
