import pytest
import numpy as np
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.postprocess_utils.roi_types import (
    SegmentationROI)
from ophys_etl.modules.segmentation.postprocess_utils.roi_merging import (
    do_rois_abut,
    merge_segmentation_rois,
    create_segmentation_roi_lookup)


@pytest.fixture
def ophys_roi_list():
    roi_list = []

    fov = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 2, 2, 2, 4, 4, 5, 5, 5, 0, 0, 0, 0],
           [0, 0, 2, 3, 3, 4, 4, 4, 8, 5, 0, 0, 0, 0],
           [0, 0, 2, 3, 3, 9, 9, 8, 8, 5, 0, 0, 0, 0],
           [0, 0, 0, 3, 3, 7, 9, 8, 8, 0, 0, 0, 0, 0],
           [0, 1, 6, 6, 7, 7, 9, 9, 0, 0, 0, 0, 0, 0],
           [0, 1, 6, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    fov = np.array(fov).astype(int)
    for ii in range(1, 10, 1):
        pixels = np.where(fov == ii)
        x0 = pixels[1].min()
        y0 = pixels[0].min()
        x1 = pixels[1].max()+1
        y1 = pixels[0].max()+1
        mask = (fov[y0:y1, x0:x1] == ii)
        roi = OphysROI(roi_id=ii,
                       x0=int(x0),
                       width=int(x1-x0),
                       y0=int(y0),
                       height=int(y1-y0),
                       valid_roi=True,
                       mask_matrix=[list(row) for row in mask])
        roi_list.append(roi)
    return roi_list


@pytest.fixture
def segmentation_roi_list(ophys_roi_list):
    roi_list = []
    for roi in ophys_roi_list:
        s_roi = SegmentationROI.from_ophys_roi(roi,
                                               flux_value=roi.roi_id)
        roi_list.append(s_roi)
    return roi_list


def compare_ophys_rois(roi0, roi1):
    assert roi0.x0 == roi1.x0
    assert roi0.y0 == roi1.y0
    assert roi0.width == roi1.width
    np.testing.assert_array_equal(roi0.mask_matrix, roi1.mask_matrix)
    assert roi0.roi_id == roi1.roi_id


def compare_segmentation_rois(roi0, roi1):
    compare_ophys_rois(roi0, roi1)
    assert np.abs(roi0.flux_value-roi1.flux_value) < 1.0e-10


def test_segmentation_roi_factory(ophys_roi_list):
    rng = np.random.RandomState(88)
    s_roi_list = []
    for roi in ophys_roi_list:
        f = rng.random_sample()
        s_roi = SegmentationROI.from_ophys_roi(roi, flux_value=f)
        s_roi_list.append(s_roi)
        compare_ophys_rois(roi, s_roi)
        assert np.abs(s_roi.flux_value-f) < 1.0e-10
        assert len(s_roi.ancestors) == 0

        # check that you can get self from get_ancestor
        anc = s_roi.get_ancestor(roi.roi_id)
        compare_segmentation_rois(anc, s_roi)

    # test ancestors
    test_roi = SegmentationROI.from_ophys_roi(ophys_roi_list[2],
                                              flux_value=55.0,
                                              ancestors=[s_roi_list[2],
                                                         s_roi_list[5]])
    compare_segmentation_rois(s_roi_list[2],
                              test_roi.get_ancestor(3))
    compare_segmentation_rois(s_roi_list[5],
                              test_roi.get_ancestor(6))

    assert test_roi.roi_id == 3
    assert np.abs(test_roi.flux_value-s_roi_list[2].flux_value) > 10.0

    with pytest.raises(RuntimeError, match='cannot get ancestor'):
        test_roi.get_ancestor(11)

    # test that the only ancestors written are the raw ancestors
    new_roi = SegmentationROI.from_ophys_roi(ophys_roi_list[7],
                                             flux_value=45.0,
                                             ancestors=[test_roi,
                                                        s_roi_list[7]])
    for ii in range(9):
        if ii in (2, 5, 7):
            compare_segmentation_rois(s_roi_list[ii],
                                      new_roi.get_ancestor(ii+1))
        else:
            with pytest.raises(RuntimeError, match='cannot get ancestor'):
                new_roi.get_ancestor(ii+1)


def test_merge_segmentation_rois(segmentation_roi_list):

    # merge all ROIs together
    new_roi = segmentation_roi_list[8]
    for ii in (7, 6, 3, 4, 2, 5, 1, 0):
        new_roi = merge_segmentation_rois(new_roi,
                                          segmentation_roi_list[ii],
                                          9,
                                          9)
    for ii in range(9):
        compare_segmentation_rois(segmentation_roi_list[ii],
                                  new_roi.get_ancestor(ii+1))

    compare_segmentation_rois(new_roi.peak,
                              segmentation_roi_list[8])

def test_create_segmentation_roi_lookup(ophys_roi_list):
    """
    really just a smoke test
    """
    rng = np.random.RandomState(1123)
    img_data = rng.normal(0.0, 0.1, (100, 100))

    # set pixels in ROI to be equal to roi_id values
    for roi in ophys_roi_list:
        x0 = roi.x0
        x1 = roi.x0+roi.width
        y0 = roi.y0
        y1 = roi.y0+roi.height
        mask = roi.mask_matrix
        img_data[y0:y1, x0:x1][mask] = float(roi.roi_id)

    lookup = create_segmentation_roi_lookup(ophys_roi_list,
                                            img_data,
                                            dx=20)
    for ii in range(1, 10, 1):
        assert np.abs(lookup[ii].flux_value-ii*10) < 5.0
