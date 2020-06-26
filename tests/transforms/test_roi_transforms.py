import pytest

import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.transforms import roi_transforms


@pytest.mark.parametrize("s2p_stat_fixture", [
    {"frame_shape": (25, 25)},
], indirect=["s2p_stat_fixture"])
def test_suite2p_rois_to_coo(s2p_stat_fixture):
    stat_path, fixture_params = s2p_stat_fixture
    frame_shape = fixture_params["frame_shape"]
    expected_rois = fixture_params["masks"]

    s2p_stat = np.load(stat_path, allow_pickle=True)
    obt_rois = roi_transforms.suite2p_rois_to_coo(s2p_stat, frame_shape)

    for obt_roi, exp_roi in zip(obt_rois, expected_rois):
        assert np.allclose(obt_roi.todense(), exp_roi)


@pytest.mark.parametrize("mask, expected, absolute_threshold, quantile", [
    # test binarize with quantile
    (np.array([[0.0, 0.5, 1.0],
               [0.0, 2.0, 2.0],
               [2.0, 1.0, 0.5]]),
     np.array([[0, 0, 1],
               [0, 1, 1],
               [1, 1, 0]]),
     None,
     0.2),

    # test binarize with absolute_threshold
    (np.array([[0.0, 0.5, 1.0],
               [0.0, 2.0, 2.0],
               [2.0, 1.0, 0.5]]),
     np.array([[0, 0, 0],
               [0, 1, 1],
               [1, 0, 0]]),
     1.5,
     None),

    # test that setting quantile will be ignored if absolute theshold is set
    (np.array([[0.0, 0.5, 1.0],
               [0.0, 2.0, 2.0],
               [2.0, 1.0, 0.5]]),
     np.array([[0, 0, 0],
               [0, 1, 1],
               [1, 0, 0]]),
     1.5,
     0.2),
])
def test_binarize_roi_mask(mask, expected, absolute_threshold, quantile):
    coo_mask = coo_matrix(mask)

    obtained = roi_transforms.binarize_roi_mask(coo_mask,
                                                absolute_threshold,
                                                quantile)

    assert np.allclose(obtained.toarray(), expected)


@pytest.mark.parametrize("mask, expected", [
    (np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]),
     (0, 0, 0, 0)),

    (np.array([[1, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]),
     (0, 1, 0, 1)),

    (np.array([[1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]),
     (0, 3, 0, 3)),

    (np.array([[1, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 1, 0],
               [0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0]]),
     (0, 5, 0, 5)),

    (np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 1, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0, 0]]),
     (1, 4, 2, 5)),
])
def test_roi_bounds(mask, expected):
    coo_mask = coo_matrix(mask)

    obtained = roi_transforms.roi_bounds(coo_mask)

    assert obtained == expected


@pytest.mark.parametrize("mask, expected, raises_error", [
    (np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]),
     np.array([[1, 1],
               [1, 1]]),
     False),

    (np.array([[0., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 2., 1., 0., 0.],
               [0., 0., 1., 1., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]]),
     np.array([[1., 0., 0.],
               [0., 2., 1.],
               [0., 1., 1.]]),
     False),

    (np.array([[1.]]),
     np.array([[1.]]),
     False),

    (np.array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]]),
     None,  # Doesn't matter what this is
     True)
])
def test_crop_roi_mask(mask, expected, raises_error):
    coo_mask = coo_matrix(mask)

    if not raises_error:
        obtained = roi_transforms.crop_roi_mask(coo_mask)
        assert np.allclose(obtained.toarray(), expected)
    else:
        with pytest.raises(ValueError, match="Cannot crop an empty ROI mask"):
            obtained = roi_transforms.crop_roi_mask(coo_mask)
