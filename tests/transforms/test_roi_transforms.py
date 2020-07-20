import pytest

import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.transforms import roi_transforms
from ophys_etl.transforms.data_loaders import motion_border


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


@pytest.mark.parametrize("coo_masks, max_correction_values, "
                         "expected, raises_error",
                         [([coo_matrix(([1, 1, 1, 1], ([0, 1, 0, 1],
                                                       [0, 0, 1, 1])),
                                       shape=(20, 20)),
                            coo_matrix(([1, 1, 1, 1], ([12, 13, 12, 13],
                                                       [12, 12, 13, 13])),
                                       shape=(20, 20))],
                           motion_border(2, 2, 2, 2),
                           [{'id': 0,
                             'cell_specimen_id': 0,
                             'x': 0,
                             'y': 0,
                             'width': 2,
                             'height': 2,
                             'valid_roi': False,
                             'mask_matrix': np.array([[True, True],
                                                      [True, True]]).tolist(),
                             'max_correction_up': 2,
                             'max_correction_down': 2,
                             'max_correction_left': 2,
                             'max_correction_right': 2,
                             'mask_image_plane': 0,
                             'exclusion_labels': [7]},
                            {'id': 1,
                             'cell_specimen_id': 1,
                             'x': 12,
                             'y': 12,
                             'width': 2,
                             'height': 2,
                             'valid_roi': True,
                             'mask_matrix': np.array([[True, True],
                                                      [True, True]]).tolist(),
                             'max_correction_up': 2,
                             'max_correction_down': 2,
                             'max_correction_left': 2,
                             'max_correction_right': 2,
                             'mask_image_plane': 0,
                             'exclusion_labels': []}],
                           False)])
def test_coo_rois_to_old(coo_masks, max_correction_values,
                         expected, raises_error):
    old_rois = roi_transforms.coo_rois_to_old(coo_masks,
                                              max_correction_values,
                                              coo_masks[0].shape)
    assert old_rois == expected


@pytest.mark.parametrize("coo_mask, expected, raises_error",
                         [(coo_matrix(([1, 1, 1, 1], ([0, 1, 0, 1],
                                                      [0, 0, 1, 1])),
                                      shape=(20, 20)),
                           {
                               'x': 0,
                               'y': 0,
                               'width': 2,
                               'height': 2,
                               'mask_matrix': np.array([[True, True],
                                                        [True, True]]
                                                       ).tolist(),
                           }, False),
                          (coo_matrix(([1, 1, 1, 1, 1, 1, 1, 1],
                                       ([0, 0, 0, 1, 1, 2, 2, 2],
                                        [0, 1, 2, 0, 2, 0, 1, 2])),
                                      shape=(20, 20)),
                           {
                               'x': 0,
                               'y': 0,
                               'width': 3,
                               'height': 3,
                               'mask_matrix': np.array([[True, True, True],
                                                        [True, False, True],
                                                        [True, True, True]]
                                                       ).tolist()
                           }, False),
                          (coo_matrix(([1, 1, 1, 1, 1],
                                       ([0, 0, 1, 1, 1],
                                        [1, 2, 0, 1, 2])),
                                      shape=(20, 20)),
                           {
                               'x': 0,
                               'y': 0,
                               'width': 3,
                               'height': 2,
                               'mask_matrix': np.array([[False, True, True],
                                                        [True, True, True]]
                                                       ).tolist()
                           }, False),
                          (coo_matrix(([1, 1, 1, 1, 1, 1, 1, 1],
                                      ([0, 0, 1, 1, 2, 2, 3, 3],
                                       [0, 1, 0, 1, 3, 4, 3, 4])),
                                      shape=(20, 20)),
                           {
                               'x': 0,
                               'y': 0,
                               'width': 5,
                               'height': 4,
                               'mask_matrix': np.array([[True, True, False,
                                                         False, False],
                                                        [True, True, False,
                                                         False, False],
                                                        [False, False, False,
                                                         True, True],
                                                        [False, False, False,
                                                         True, True]]).tolist()
                           }, False),
                          (coo_matrix(([1, 1, 1, 1, 1],
                                       ([3, 4, 4, 5, 5],
                                        [3, 3, 4, 3, 4])),
                                      shape=(20, 20)),
                           {
                               'x': 3,
                               'y': 3,
                               'width': 2,
                               'height': 3,
                               'mask_matrix': np.array([[True, False],
                                                        [True, True],
                                                        [True, True]]).tolist()
                           }, False)])
def test_coo_mask_to_old_format(coo_mask, expected, raises_error):
    old_style_roi = roi_transforms._coo_mask_to_old_format(coo_mask)
    assert old_style_roi == expected


@pytest.mark.parametrize("old_type_roi, movie_shape, expected_label, "
                         "expected_valid, raises_error",
                         [({'x': 10, 'y': 10, 'width': 2,
                            'height': 2, 'valid_roi': True,
                            'max_correction_up': 2,
                            'max_correction_down': 2, 'max_correction_left': 2,
                            'max_correction_right': 2, 'exclusion_labels': []},
                          (20, 20), [], True, False),
                          ({'x': 10, 'y': 10, 'width': 2,
                            'height': 2, 'valid_roi': True,
                            'max_correction_up': 15,
                            'max_correction_down': 2, 'max_correction_left': 2,
                            'max_correction_right': 2, 'exclusion_labels': []},
                          (20, 20), [7], False, False),
                          ({'x': 10, 'y': 10, 'width': 2,
                            'height': 2, 'valid_roi': True,
                            'max_correction_up': 2,
                            'max_correction_down': 15,
                            'max_correction_left': 2,
                            'max_correction_right': 2, 'exclusion_labels': []},
                           (20, 20), [7], False, False),
                          ({'x': 10, 'y': 10, 'width': 2,
                            'height': 2, 'valid_roi': True,
                            'max_correction_up': 2,
                            'max_correction_down': 2,
                            'max_correction_left': 15,
                            'max_correction_right': 2, 'exclusion_labels': []},
                          (20, 20), [7], False, False),
                          ({'x': 10, 'y': 10, 'width': 2,
                            'height': 2, 'valid_roi': True,
                            'max_correction_up': 2,
                            'max_correction_down': 2, 'max_correction_left': 2,
                            'max_correction_right': 15, 'exclusion_labels': []
                            },
                          (20, 20), [7], False, False),
                          ({'x': 10, 'y': 10, 'width': 2,
                            'height': 2, 'valid_roi': True,
                            'max_correction_up': 10,
                            'max_correction_down': 2, 'max_correction_left': 2,
                            'max_correction_right': 2, 'exclusion_labels': []},
                          (20, 20), [7], False, False),
                          ({'x': 10, 'y': 10, 'width': 2,
                            'height': 2, 'valid_roi': True,
                            'max_correction_up': 2,
                            'max_correction_down': 9, 'max_correction_left': 2,
                            'max_correction_right': 2, 'exclusion_labels': []},
                          (20, 20), [7], False, False)
                          ])
def test_check_motion_exclusion(old_type_roi, movie_shape, expected_label,
                                expected_valid, raises_error):
    """
    Test Cases:
    1: ROI roughly centered in frame and not out of bounds
    2: ROI fully out of motion correction bound in up direction
    3: ROI fully out of motion correction bound in down direction
    4: ROI fully out of motion correction bound in left direction
    5: ROI fully out of motion correction bound in right direction
    6: ROI edge on the motion correction border in the up direction
    7: ROI halfway in / out of motion correction border in down direction
    """
    roi_transforms._check_motion_exclusion(old_type_roi, movie_shape)
    assert old_type_roi['exclusion_labels'] == expected_label
    assert old_type_roi['valid_roi'] == expected_valid
