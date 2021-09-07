import pytest
import h5py
import copy
import numpy as np

from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.utils.roi_utils import (
    merge_rois,
    do_rois_abut,
    extract_roi_to_ophys_roi,
    ophys_roi_to_extract_roi,
    sub_video_from_roi,
    intersection_over_union,
    convert_to_lims_roi,
    ophys_roi_list_from_file,
    serialize_extract_roi_list,
    deserialize_extract_roi_list,
    check_matching_extract_roi_lists,
    select_contiguous_region,
    background_mask_from_roi_list,
    select_window_from_background,
    pixel_list_to_extract_roi)


@pytest.fixture
def extract_roi_list():
    nroi = 20
    xs = np.linspace(start=10, stop=490, num=nroi, dtype=int)
    ys = np.linspace(start=20, stop=470, num=nroi, dtype=int)
    widths = ([10] * (nroi - 10) + [12] * 10)
    heights = ([8] * (nroi - 7) + [11] * 7)
    valids = ([True] * (nroi - 4) + [False] * 4)

    rng = np.random.default_rng(3342)
    roi_list = []
    for i in range(nroi):
        mask = [row.tolist()
                for row in rng.integers(low=0,
                                        high=2,
                                        size=(heights[i], widths[i])
                                        ).astype(bool)]
        roi_list.append(
                ExtractROI(
                    id=int((i + 1)),
                    x=int(xs[i]),
                    y=int(ys[i]),
                    width=int(widths[i]),
                    height=int(heights[i]),
                    mask=mask,
                    valid=valids[i]))
    return roi_list


@pytest.mark.parametrize("to_hdf5", [True, False])
def test_serialization(extract_roi_list, to_hdf5, tmpdir):
    serialized = serialize_extract_roi_list(extract_roi_list)

    if to_hdf5:
        # check round trip to hdf5 dataset
        h5path = tmpdir / "test.h5"
        with h5py.File(h5path, "w") as f:
            f.create_dataset("test_rois", data=serialized)
        with h5py.File(h5path, "r") as f:
            deserialized = deserialize_extract_roi_list(
                    f["test_rois"][()])
    else:
        # check independently of hdf5
        deserialized = deserialize_extract_roi_list(serialized)
    for round_tripped, original in zip(deserialized, extract_roi_list):
        assert round_tripped == original


def test_check_matching_extract_roi_lists(extract_roi_list):
    # straight copy, should match
    listB = copy.deepcopy(extract_roi_list)
    check_matching_extract_roi_lists(extract_roi_list, listB)

    # different order, should match
    check_matching_extract_roi_lists(extract_roi_list, listB[::-1])

    # missing ROI, should not match
    listB.pop()
    with pytest.raises(AssertionError,
                       match=r"ids in ROI lists do not match"):
        check_matching_extract_roi_lists(extract_roi_list, listB[::-1])

    # differing ROIs, should not match
    listB = copy.deepcopy(extract_roi_list)
    listB[0]["x"] += 2
    with pytest.raises(AssertionError,
                       match=f"roi with ID {listB[0]['id']} does not match"):
        check_matching_extract_roi_lists(extract_roi_list, listB[::-1])


@pytest.mark.parametrize(
    "origin,mask,expected",
    [
     ((14, 22), np.array([[False, False, False, False, False],
                          [False, True, False, False, False],
                          [False, False, True, False, False],
                          [False, False, False, False, False],
                          [False, False, False, False, False]]),
      ExtractROI(
          id=0,
          x=23,
          y=15,
          width=2,
          height=2,
          mask=[[True, False], [False, True]],
          valid=True)
      )
    ])
def test_roi_converter(origin, mask, expected):
    """
    Test method that converts ROIs to LIMS-like ROIs
    """
    actual = convert_to_lims_roi(origin, mask)
    assert actual == expected


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

    ophys_roi = extract_roi_to_ophys_roi(roi)
    assert ophys_roi.x0 == roi['x']
    assert ophys_roi.y0 == roi['y']
    assert ophys_roi.height == roi['height']
    assert ophys_roi.width == roi['width']
    assert ophys_roi.roi_id == roi['id']
    assert ophys_roi.valid_roi and roi['valid']
    np.testing.assert_array_equal(ophys_roi.mask_matrix, mask)


def test_ophys_roi_to_extract_roi(example_roi_list):
    for roi_in in example_roi_list:
        roi_out = ophys_roi_to_extract_roi(roi_in)
        assert roi_out['x'] == roi_in.x0
        assert roi_out['y'] == roi_in.y0
        assert roi_out['width'] == roi_in.width
        assert roi_out['height'] == roi_in.height
        assert roi_out['id'] == roi_in.roi_id
        np.testing.assert_array_equal(roi_in.mask_matrix,
                                      roi_out['mask'])


def test_merge_rois():

    x0 = 11
    y0 = 22
    height = 5
    width = 6
    mask = np.zeros((height, width), dtype=bool)
    mask[4, 5] = True
    mask[3, 5] = True
    mask[0, 0] = True
    mask[1, 4] = True
    roi0 = OphysROI(x0=x0,
                    width=width,
                    y0=y0,
                    height=height,
                    mask_matrix=mask,
                    roi_id=0,
                    valid_roi=True)

    y0 = 19
    x0 = 16
    height = 6
    width = 4
    mask = np.zeros((height, width), dtype=bool)
    mask[5, 0] = True
    mask[5, 1] = True
    mask[5, 2] = True
    mask[3, 3] = True

    roi1 = OphysROI(x0=x0,
                    y0=y0,
                    width=width,
                    height=height,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    merged_roi = merge_rois(roi0, roi1, 2)
    assert merged_roi.roi_id == 2
    assert merged_roi.x0 == 11
    assert merged_roi.y0 == 19
    assert merged_roi.width == 9
    assert merged_roi.height == 8

    # make sure all pixels that should be marked
    # True are
    true_pix = set()
    new_mask = merged_roi.mask_matrix
    for roi in (roi0, roi1):
        x0 = roi.x0
        y0 = roi.y0
        mask = roi.mask_matrix
        for ir in range(roi.height):
            for ic in range(roi.width):
                if not mask[ir, ic]:
                    continue
                row = ir+y0-merged_roi.y0
                col = ic+x0-merged_roi.x0
                assert new_mask[row, col]
                true_pix.add((row, col))
    # make sure no extraneous pixels were marked True
    assert len(true_pix) == new_mask.sum()


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

    assert do_rois_abut(roi0, roi1, pixel_distance=1.0)

    # just touching
    roi1 = OphysROI(x0=26,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert do_rois_abut(roi0, roi1, pixel_distance=1.0)

    roi1 = OphysROI(x0=27,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert not do_rois_abut(roi0, roi1, pixel_distance=1.0)

    # they are, however, just diagonally 1 pixel away
    # from each other
    assert do_rois_abut(roi0, roi1, pixel_distance=np.sqrt(2))

    # gap of one pixel
    assert do_rois_abut(roi0, roi1, pixel_distance=2)

    roi1 = OphysROI(x0=28,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert not do_rois_abut(roi0, roi1, pixel_distance=2)


def test_sub_video_from_roi(example_roi0):
    rng = np.random.RandomState(51433)
    video_data = rng.randint(11, 457, (100, 50, 47))

    sub_video = sub_video_from_roi(example_roi0, video_data)

    npix = example_roi0.mask_matrix.sum()
    expected = np.zeros((100, npix), dtype=int)
    mask = example_roi0.mask_matrix
    i_pix = 0
    for ir in range(example_roi0.height):
        for ic in range(example_roi0.width):
            if not mask[ir, ic]:
                continue
            expected[:, i_pix] = video_data[:,
                                            example_roi0.y0+ir,
                                            example_roi0.x0+ic]
            i_pix += 1
    np.testing.assert_array_equal(expected, sub_video)


def test_intersection_over_union():

    width = 7
    height = 5
    mask = np.ones((height, width), dtype=bool)
    mask[:, 4:] = False
    roi0 = OphysROI(roi_id=0,
                    x0=100,
                    y0=200,
                    width=width,
                    height=height,
                    valid_roi=True,
                    mask_matrix=mask)

    width = 4
    height = 9
    mask = np.ones((height, width), dtype=bool)
    mask[:, 0] = False
    mask[2:, :] = False
    roi1 = OphysROI(roi_id=0,
                    x0=101,
                    y0=201,
                    width=width,
                    height=height,
                    valid_roi=True,
                    mask_matrix=mask)

    # expected_intersection = 4
    # expected_union = 22

    expected = 4.0/22.0

    actual = intersection_over_union(roi0, roi1)
    actual1 = intersection_over_union(roi1, roi0)
    eps = 1.0e-20
    np.testing.assert_allclose(actual, actual1, rtol=0.0, atol=eps)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=eps)


def test_ophys_roi_list_from_file(roi_file, list_of_roi):
    raw_actual = ophys_roi_list_from_file(roi_file)
    actual = [ophys_roi_to_extract_roi(roi)
              for roi in raw_actual]
    assert actual == list_of_roi


def test_select_contiguous_region():

    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    mask[3:10, 7:10] = True

    # check that correct error is raised when you
    # pass in invalid seed_pt
    with pytest.raises(
            IndexError,
            match='does not exist in mask with shape \\(10, 10\\)'):
        select_contiguous_region((100, 100), mask)

    with pytest.raises(
            IndexError,
            match='does not exist in mask with shape \\(10, 10\\)'):
        select_contiguous_region((100, 3), mask)

    with pytest.raises(
            IndexError,
            match='does not exist in mask with shape \\(10, 10\\)'):
        select_contiguous_region((3, 100), mask)

    output = select_contiguous_region((3, 3), mask)
    expected = np.zeros((10, 10), dtype=bool)
    expected[2:5, 2:5] = True
    np.testing.assert_array_equal(output, expected)

    output = select_contiguous_region((4, 9), mask)
    expected = np.zeros((10, 10), dtype=bool)
    expected[3:10, 7:10] = True
    np.testing.assert_array_equal(output, expected)

    # try when seed_pt is not True
    output = select_contiguous_region((2, 9), mask)
    expected = np.zeros((10, 10), dtype=bool)
    np.testing.assert_array_equal(output, expected)

    # try with diagonally connected blocks
    mask[1, 1] = True
    output = select_contiguous_region((3, 3), mask)
    expected = np.zeros((10, 10), dtype=bool)
    expected[2:5, 2:5] = True
    expected[1, 1] = True
    np.testing.assert_array_equal(output, expected)


def test_background_mask_from_roi_list():
    roi_list = []
    mask = [[True, True, False], [False, False, True]]
    roi = OphysROI(x0=2, width=3,
                   y0=1, height=2,
                   roi_id=0, valid_roi=True,
                   mask_matrix=mask)
    roi_list.append(roi)

    mask = [[True, False], [False, False], [True, True]]
    roi = OphysROI(x0=3, width=2,
                   y0=0, height=3,
                   roi_id=1, valid_roi=True,
                   mask_matrix=mask)

    roi_list.append(roi)

    mask = [[True, False], [False, False], [True, True]]
    roi = OphysROI(x0=0, width=2,
                   y0=1, height=3,
                   roi_id=2, valid_roi=True,
                   mask_matrix=mask)

    roi_list.append(roi)

    img_shape = (4, 6)

    expected = [[True, True, True, False, True, True],
                [False, True, False, False, True, True],
                [True, True, True, False, False, True],
                [False, False, True, True, True, True]]
    expected = np.array(expected)
    actual = background_mask_from_roi_list(roi_list, img_shape)
    np.testing.assert_array_equal(actual, expected)

    # assert an error gets raised if img_shape is too small
    # (this actually relies on native numpy IndexErrors being
    # raised; I just want to be aware if numpy changes its
    # default behavior out from under us)
    with pytest.raises(Exception):
        background_mask_from_roi_list(roi_list, (4, 3))


def test_select_window_from_background():

    # 9 x 8
    background = [[0, 0, 1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1],
                  [1, 1, 0, 0, 0, 1, 1, 1],
                  [1, 1, 0, 0, 0, 1, 1, 1],
                  [1, 1, 0, 0, 0, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1]]
    background = np.array(background)

    roi = OphysROI(x0=2, width=3,
                   y0=2, height=3,
                   valid_roi=True, roi_id=0,
                   mask_matrix=[[True, True, True],
                                [True, True, True],
                                [True, True, True]])

    # check when you ask for too much
    ((rowmin, rowmax),
     (colmin, colmax)) = select_window_from_background(
                             roi,
                             background,
                             60)
    assert rowmin == 0
    assert rowmax == 9
    assert colmin == 0
    assert colmax == 8

    # should just get one pixel outside of roi
    ((rowmin, rowmax),
     (colmin, colmax)) = select_window_from_background(
                             roi,
                             background,
                             15)
    assert rowmin == 1
    assert rowmax == 6
    assert colmin == 1
    assert colmax == 6

    # should get two pixels outside of roi
    ((rowmin, rowmax),
     (colmin, colmax)) = select_window_from_background(
                             roi,
                             background,
                             16)

    assert rowmin == 0
    assert rowmax == 7
    assert colmin == 0
    assert colmax == 7

    # should get three pixels outside of roi
    ((rowmin, rowmax),
     (colmin, colmax)) = select_window_from_background(
                             roi,
                             background,
                             37)

    assert rowmin == 0
    assert rowmax == 8
    assert colmin == 0
    assert colmax == 8

    # move to the corner ROI
    roi = OphysROI(x0=0, width=2,
                   y0=0, height=2,
                   valid_roi=True, roi_id=0,
                   mask_matrix=[[True, True],
                                [True, True]])

    ((rowmin, rowmax),
     (colmin, colmax)) = select_window_from_background(
                             roi,
                             background,
                             5)

    assert rowmin == 0
    assert rowmax == 4
    assert colmin == 0
    assert colmax == 4


def test_pixel_list_to_extract_roi():
    pixel_list = [(1, 2), (4, 7), (9, 2)]
    roi = pixel_list_to_extract_roi(pixel_list, 5)
    assert roi['id'] == 5
    assert roi['x'] == 2
    assert roi['y'] == 1
    assert roi['width'] == 6
    assert roi['height'] == 9

    assert len(roi['mask']) == roi['height']
    for row in roi['mask']:
        assert len(row) == roi['width']

    ophys_roi = extract_roi_to_ophys_roi(roi)
    for p in pixel_list:
        assert p in ophys_roi.global_pixel_set
    assert len(pixel_list) == len(ophys_roi.global_pixel_set)
