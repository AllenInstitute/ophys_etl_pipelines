import pytest

import numpy as np
import copy
from itertools import product
from scipy.sparse import coo_matrix

from ophys_etl.utils.motion_border import MotionBorder
from ophys_etl.utils import rois as rois_utils
from ophys_etl.types import DenseROI, OphysROI
from ophys_etl.schemas import DenseROISchema


@pytest.fixture
def example_ophys_roi_list_fixture():
    rng = np.random.RandomState(6412439)
    roi_list = []
    for ii in range(30):
        x0 = rng.randint(0, 25)
        y0 = rng.randint(0, 25)
        height = rng.randint(3, 7)
        width = rng.randint(3, 7)
        mask = rng.randint(0, 2, (height, width)).astype(bool)
        roi = OphysROI(x0=x0, y0=y0,
                       height=height, width=width,
                       mask_matrix=mask,
                       roi_id=ii,
                       valid_roi=True)
        roi_list.append(roi)

    return roi_list


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
            rois_utils.full_mask_constructor(mask_matrix, x, y, shape)
    else:
        full_mask = rois_utils.full_mask_constructor(mask_matrix,
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
    new_roi = rois_utils.roi_from_full_mask(roi, full_mask)
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
        assert rois_utils.morphological_transform(d, (50, 50)) is None
        return

    morphed = rois_utils.morphological_transform(d, (50, 50))
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
    e = rois_utils.dense_to_extract(d)
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
    obt_rois = rois_utils.suite2p_rois_to_coo(s2p_stat, frame_shape)

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

    obtained = rois_utils.binarize_roi_mask(coo_mask,
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

    obtained = rois_utils.roi_bounds(coo_mask)

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

    obtained = rois_utils.crop_roi_mask(coo_mask)
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
    roi = rois_utils._coo_mask_to_LIMS_compatible_format(coo)
    roi['max_correction_up'] = max_correction_vals.up
    roi['max_correction_down'] = max_correction_vals.down
    roi['max_correction_right'] = max_correction_vals.right
    roi['max_correction_left'] = max_correction_vals.left

    valid = rois_utils._motion_exclusion(roi, coo.shape)

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
    roi = rois_utils._coo_mask_to_LIMS_compatible_format(coo)

    valid = rois_utils._small_size_exclusion(roi, npixel_threshold)

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
    compatible_rois = rois_utils.coo_rois_to_lims_compatible(
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
    roi = rois_utils._coo_mask_to_LIMS_compatible_format(coo_mask)
    for k in expected:
        assert roi[k] == expected[k]


def test_roi_abut():

    height = 6
    width = 7
    mask = np.zeros((height, width), dtype=bool)
    mask[1:5, 1:6] = True

    # overlapping
    roi0 = OphysROI(x0=22,
                    y0=44,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=0,
                    valid_roi=True)

    roi1 = OphysROI(x0=23,
                    y0=46,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert rois_utils.do_rois_abut(roi0, roi1, pixel_distance=1.0)

    # just touching
    roi1 = OphysROI(x0=26,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert rois_utils.do_rois_abut(roi0, roi1, pixel_distance=1.0)

    roi1 = OphysROI(x0=27,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert not rois_utils.do_rois_abut(roi0, roi1, pixel_distance=1.0)

    # they are, however, just diagonally 1 pixel away
    # from each other
    assert rois_utils.do_rois_abut(roi0, roi1, pixel_distance=np.sqrt(2))

    # gap of one pixel
    assert rois_utils.do_rois_abut(roi0, roi1, pixel_distance=2)

    roi1 = OphysROI(x0=28,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert not rois_utils.do_rois_abut(roi0, roi1, pixel_distance=2)


def test_extract_roi_to_ophys_roi():
    rng = np.random.RandomState(345)
    mask = rng.randint(0, 2, (9, 7)).astype(bool)
    roi = {'x': 5,
           'y': 6,
           'width': 7,
           'height': 9,
           'id': 991,
           'valid': True,
           'mask': [list(i) for i in mask]}

    ophys_roi = rois_utils.extract_roi_to_ophys_roi(roi)
    assert ophys_roi.x0 == roi['x']
    assert ophys_roi.y0 == roi['y']
    assert ophys_roi.height == roi['height']
    assert ophys_roi.width == roi['width']
    assert ophys_roi.roi_id == roi['id']
    assert ophys_roi.valid_roi and roi['valid']
    np.testing.assert_array_equal(ophys_roi.mask_matrix, mask)


def test_ophys_roi_to_extract_roi(example_ophys_roi_list_fixture):
    for roi_in in example_ophys_roi_list_fixture:
        roi_out = rois_utils.ophys_roi_to_extract_roi(roi_in)
        assert roi_out['x'] == roi_in.x0
        assert roi_out['y'] == roi_in.y0
        assert roi_out['width'] == roi_in.width
        assert roi_out['height'] == roi_in.height
        assert roi_out['id'] == roi_in.roi_id
        np.testing.assert_array_equal(roi_in.mask_matrix,
                                      roi_out['mask'])


@pytest.mark.parametrize(
    "munge_mapping, extra_field",
    product([{'valid': 'valid_roi'},
             {'mask': 'mask_matrix'},
             {'id': 'roi_id'},
             {'valid': 'valid_roi', 'mask': 'mask_matrix'},
             {'valid': 'valid_roi', 'id': 'roi_id'},
             {'mask': 'mask_matrix', 'id': 'roi_id'},
             {'valid': 'valid_roi', 'mask': 'mask_matrix', 'id': 'roi_id'}],
            (True, False))
)
def test_sanitize_extract_roi_list(
        example_ophys_roi_list_fixture,
        munge_mapping,
        extra_field):

    extract_roi_list = [rois_utils.ophys_roi_to_extract_roi(roi)
                        for roi in example_ophys_roi_list_fixture]

    if extra_field:
        for ii, roi in enumerate(extract_roi_list):
            roi['nonsense'] = ii
        for roi in extract_roi_list:
            assert 'nonsense' in roi

    bad_roi_list = []
    for roi in extract_roi_list:
        new_roi = copy.deepcopy(roi)
        for k in munge_mapping:
            new_roi[munge_mapping[k]] = new_roi.pop(k)
        bad_roi_list.append(new_roi)

    assert bad_roi_list != extract_roi_list
    cleaned_roi_list = rois_utils.sanitize_extract_roi_list(bad_roi_list)
    assert cleaned_roi_list == extract_roi_list
