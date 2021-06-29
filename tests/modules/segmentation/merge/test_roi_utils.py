import pytest
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.merge.roi_utils import (
    merge_rois,
    do_rois_abut,
    extract_roi_to_ophys_roi,
    ophys_roi_to_extract_roi)


@pytest.fixture
def example_roi_list():
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

    assert do_rois_abut(roi0, roi1, dpix=1.0)

    # just touching
    roi1 = OphysROI(x0=26,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert do_rois_abut(roi0, roi1, dpix=1.0)

    roi1 = OphysROI(x0=27,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert not do_rois_abut(roi0, roi1, dpix=1.0)

    # they are, however, just diagonally 1 pixel away
    # from each other
    assert do_rois_abut(roi0, roi1, dpix=np.sqrt(2))

    # gap of one pixel
    assert do_rois_abut(roi0, roi1, dpix=2)

    roi1 = OphysROI(x0=28,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert not do_rois_abut(roi0, roi1, dpix=2)
