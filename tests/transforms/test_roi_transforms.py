import pytest

import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.extractors.motion_correction import MotionBorder
from ophys_etl.transforms import roi_transforms
from ophys_etl.types import DenseROI
from ophys_etl.schemas import DenseROISchema


@pytest.mark.parametrize(
    "np_mask_matrix, x, y, shape, expected, fail",
    [
        (
            np.array([[0, 0, 1, 0],
                      [0, 1, 1, 0],
                      [0, 1, 1, 0],
                      [0, 1, 1, 1]]).astype(bool),
            2, 3, (7, 7),
            np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 1, 1, 0, 0],
                      [0, 0, 0, 1, 1, 1, 0]]).astype(bool),
            False,
        ),
        (
            np.array([[0, 0, 1, 0],
                      [0, 1, 1, 0],
                      [0, 1, 1, 0],
                      [0, 1, 1, 1]]).astype(bool),
            2, 3, (5, 5), None, True
        ),
    ]
)
def test_full_mask_constructor(np_mask_matrix, x, y, shape, expected, fail):
    mask_matrix = [i.tolist() for i in np_mask_matrix]
    if fail:
        with pytest.raises(ValueError,
                           match=r"index can't contain negative values"):
            roi_transforms.full_mask_constructor(mask_matrix, x, y, shape)
    else:
        full_mask = roi_transforms.full_mask_constructor(mask_matrix,
                                                         x, y, shape)
        np.testing.assert_array_equal(full_mask, expected)


@pytest.mark.parametrize(
    "full_mask, roi, expected",
    [
        (
            np.array([[0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 1, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0]]).astype(bool),
            {
                'extra key': "should propogate"
            },
            {
                'x': 1,
                'y': 2,
                'width': 3,
                'height': 4,
                'extra key': "should propogate",
                'mask_matrix': [[True, False, False],
                                [True, True, False],
                                [False, True, False],
                                [False, True, True]]
            }
        )
    ]
)
def test_roi_from_full_mask(full_mask, roi, expected):
    new_roi = roi_transforms.roi_from_full_mask(roi, full_mask)
    assert new_roi == expected


@pytest.mark.parametrize(
    "mask_matrix, expect_None",
    [
        (
            [[True, True], [False, True]],
            True,
        ),
        (
            [[True, True, True, True],
             [True, True, True, True],
             [True, True, True, True],
             [True, True, True, True]],
            False
        )
    ]
)
def test_morphological_transform(mask_matrix, expect_None):
    d = DenseROI(id=1, x=23, y=34, width=128, height=128, valid_roi=True,
                 mask_matrix=mask_matrix,
                 max_correction_up=12,
                 max_correction_down=12,
                 max_correction_left=12,
                 max_correction_right=12,
                 mask_image_plane=0,
                 exclusion_labels=['small_size', 'motion_border'])
    if expect_None:
        assert roi_transforms.morphological_transform(d, (50, 50)) is None
        return

    morphed = roi_transforms.morphological_transform(d, (50, 50))
    DenseROISchema().validate(morphed)

    for k in ['id', 'valid_roi', 'mask_image_plane', 'exclusion_labels']:
        assert morphed[k] == d[k]
    for k in d.keys():
        if 'max_correction' in k:
            assert morphed[k] == d[k]


def test_dense_to_extract():
    d = DenseROI(
            id=1,
            x=23,
            y=34,
            width=128,
            height=128,
            valid_roi=True,
            mask_matrix=[[True, True], [False, True]],
            max_correction_up=12,
            max_correction_down=12,
            max_correction_left=12,
            max_correction_right=12,
            mask_image_plane=0,
            exclusion_labels=['small_size', 'motion_border'])
    e = roi_transforms.dense_to_extract(d)
    for k in ['id', 'x', 'y', 'width', 'height']:
        assert e[k] == d[k]
    assert e['mask'] == d['mask_matrix']
    assert e['valid'] == d['valid_roi']


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
     None),

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


@pytest.mark.parametrize("mask, expected", [
    (np.array([[0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0]]),
     np.array([[1, 1],
               [1, 1]])),
    (np.array([[0., 0., 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 2., 1., 0., 0.],
               [0., 0., 1., 1., 0., 0.],
               [0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0., 0.]]),
     np.array([[1., 0., 0.],
               [0., 2., 1.],
               [0., 1., 1.]])),
    (np.array([[1.]]),
     np.array([[1.]])),
    (np.array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]]),
     None)
])
def test_crop_roi_mask(mask, expected):
    coo_mask = coo_matrix(mask)

    obtained = roi_transforms.crop_roi_mask(coo_mask)
    if expected is None:
        assert obtained is None
    else:
        assert np.allclose(obtained.toarray(), expected)


@pytest.mark.parametrize(
        "dense_mask, max_correction_vals, expected",
        # MotionBorder in order of: [left, right, up, down]
        [
            ([[1]], MotionBorder(0, 0, 0, 0), True),
            ([[1]], MotionBorder(1, 0, 0, 0), False),
            ([[1]], MotionBorder(0, 1, 0, 0), False),
            ([[1]], MotionBorder(0, 0, 1, 0), False),
            ([[1]], MotionBorder(0, 0, 0, 1), False),
            ([[1]], MotionBorder(0.2, 0, 0, 0), False),
            ([[1]], MotionBorder(0, 0.2, 0, 0), False),
            ([[1]], MotionBorder(0, 0, 0.2, 0), False),
            ([[1]], MotionBorder(0, 0, 0, 0.2), False),
            (
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 0, 0]], MotionBorder(1, 1, 1, 1), True),
            (
                [[0, 1, 0],
                 [0, 1, 0],
                 [0, 0, 0]], MotionBorder(1, 1, 1, 0), True),
            (
                [[0, 1, 0],
                 [0, 1, 0],
                 [0, 0, 0]], MotionBorder(1, 1, 1, 1), False),
            (
                [[0, 1, 0],
                 [0, 1, 0],
                 [0, 0, 0]], MotionBorder(1, 1, 1, 0.2), False),
            (
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 1, 0]], MotionBorder(1, 1, 0, 1), True),
            (
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 1, 0]], MotionBorder(1, 1, 1, 1), False),
            (
                [[0, 0, 0],
                 [0, 1, 0],
                 [0, 1, 0]], MotionBorder(1, 1, 0.2, 1), False),
            (
                [[0, 0, 0],
                 [1, 1, 0],
                 [0, 0, 0]], MotionBorder(1, 0, 1, 1), True),
            (
                [[0, 0, 0],
                 [1, 1, 0],
                 [0, 0, 0]], MotionBorder(1, 1, 1, 1), False),
            (
                [[0, 0, 0],
                 [1, 1, 0],
                 [0, 0, 0]], MotionBorder(1, 0.2, 1, 1), False),
            (
                [[0, 0, 0],
                 [0, 1, 1],
                 [0, 0, 0]], MotionBorder(0, 1, 1, 1), True),
            (
                [[0, 0, 0],
                 [0, 1, 1],
                 [0, 0, 0]], MotionBorder(1, 1, 1, 1), False),
            (
                [[0, 0, 0],
                 [0, 1, 1],
                 [0, 0, 0]], MotionBorder(0.2, 1, 1, 1), False),
            (
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]], MotionBorder(2, 2, 2, 2), True),
            (
                [[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 1, 1, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]], MotionBorder(2, 3, 2, 2), False),
                ])
def test_motion_exclusion(dense_mask, max_correction_vals, expected):
    coo = coo_matrix(dense_mask)
    roi = roi_transforms._coo_mask_to_LIMS_compatible_format(coo)
    roi['max_correction_up'] = max_correction_vals.up
    roi['max_correction_down'] = max_correction_vals.down
    roi['max_correction_right'] = max_correction_vals.right
    roi['max_correction_left'] = max_correction_vals.left

    valid = roi_transforms._motion_exclusion(roi, coo.shape)

    assert valid == expected


@pytest.mark.parametrize(
        "dense_mask, npixel_threshold, expected",
        [
            ([[1]], 0, True),
            ([[1]], 1, False),
            ([[0, 0, 0],
              [0, 1, 0],
              [0, 0, 0]], 4, False),
            ([[0, 0, 0],
              [0, 1, 1],
              [0, 1, 1]], 4, False),
            ([[0, 0, 0],
              [1, 1, 1],
              [0, 1, 1]], 4, True)
            ])
def test_small_size_exclusion(dense_mask, npixel_threshold, expected):
    coo = coo_matrix(dense_mask)
    roi = roi_transforms._coo_mask_to_LIMS_compatible_format(coo)

    valid = roi_transforms._small_size_exclusion(roi, npixel_threshold)

    assert valid == expected


@pytest.mark.parametrize(
        "coo_masks, max_correction_values, npixel_threshold, expected",
        [([
            coo_matrix(([1, 1, 1, 1], ([0, 1, 0, 1], [0, 0, 1, 1])),
                       shape=(20, 20)),
            coo_matrix(([1, 1, 1, 0], ([12, 13, 12, 13], [12, 12, 13, 13])),
                       shape=(20, 20)),
            # this empty one should get filtered away
            coo_matrix([[]])
           ],
          MotionBorder(2.5, 2.5, 2.5, 2.5), 3,
          [{'id': 0,
            'x': 0,
            'y': 0,
            'width': 2,
            'height': 2,
            'valid_roi': False,
            'mask_matrix': np.array([[True, True],
                                     [True, True]]).tolist(),
            'max_correction_up': 2.5,
            'max_correction_down': 2.5,
            'max_correction_left': 2.5,
            'max_correction_right': 2.5,
            'mask_image_plane': 0,
            'exclusion_labels': ['motion_border']},
           {'id': 1,
            'x': 12,
            'y': 12,
            'width': 2,
            'height': 2,
            'valid_roi': False,
            'mask_matrix': np.array([[True, True],
                                     [True, False]]).tolist(),
            'max_correction_up': 2.5,
            'max_correction_down': 2.5,
            'max_correction_left': 2.5,
            'max_correction_right': 2.5,
            'mask_image_plane': 0,
            'exclusion_labels': ['small_size']}
           ])])
def test_coo_rois_to_compatible(coo_masks, max_correction_values,
                                npixel_threshold, expected):
    compatible_rois = roi_transforms.coo_rois_to_lims_compatible(
        coo_masks,
        max_correction_values,
        coo_masks[0].shape,
        npixel_threshold=npixel_threshold)
    assert compatible_rois == expected


@pytest.mark.parametrize("coo_mask, expected",
                         [(coo_matrix(([1, 1, 1, 1], ([0, 1, 0, 1],
                                                      [0, 0, 1, 1])),
                                      shape=(20, 20)),
                           {
                               'x': 0,
                               'y': 0,
                               'width': 2,
                               'height': 2,
                               'mask_matrix': [[True, True],
                                               [True, True]]}),
                          (coo_matrix(([1, 1, 1, 1, 1, 1, 1, 1],
                                       ([0, 0, 0, 1, 1, 2, 2, 2],
                                        [0, 1, 2, 0, 2, 0, 1, 2])),
                                      shape=(20, 20)),
                           {
                               'x': 0,
                               'y': 0,
                               'width': 3,
                               'height': 3,
                               'mask_matrix': [[True, True, True],
                                               [True, False, True],
                                               [True, True, True]]}),
                          (coo_matrix(([1, 1, 1, 1, 1],
                                       ([0, 0, 1, 1, 1],
                                        [1, 2, 0, 1, 2])),
                                      shape=(20, 20)),
                           {
                               'x': 0,
                               'y': 0,
                               'width': 3,
                               'height': 2,
                               'mask_matrix': [[False, True, True],
                                               [True, True, True]]}),
                          (coo_matrix(([1, 1, 1, 1, 1, 1, 1, 1],
                                      ([0, 0, 1, 1, 2, 2, 3, 3],
                                       [0, 1, 0, 1, 3, 4, 3, 4])),
                                      shape=(20, 20)),
                           {
                               'x': 0,
                               'y': 0,
                               'width': 5,
                               'height': 4,
                               'mask_matrix': [[True, True, False, False,
                                                False],
                                               [True, True, False, False,
                                                False],
                                               [False, False, False, True,
                                                True],
                                               [False, False, False, True,
                                                True]]}),
                          (coo_matrix(([1, 1, 1, 1, 1],
                                       ([3, 4, 4, 5, 5],
                                        [3, 3, 4, 3, 4])),
                                      shape=(20, 20)),
                           {
                               'x': 3,
                               'y': 3,
                               'width': 2,
                               'height': 3,
                               'mask_matrix': [[True, False],
                                               [True, True],
                                               [True, True]]}),
                          (coo_matrix(([1, 1, 1],
                                       ([0, 0, 1],
                                        [1, 2, 2])),
                                      shape=(5, 5)),
                           {
                               'x': 1,
                               'y': 0,
                               'width': 2,
                               'height': 2,
                               'mask_matrix': [[True, True],
                                               [False, True]]})
                          ])
def test_coo_mask_to_compatible_format(coo_mask, expected):
    roi = roi_transforms._coo_mask_to_LIMS_compatible_format(coo_mask)
    for k in expected:
        assert roi[k] == expected[k]
