import pytest
import numpy as np
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.postprocess_utils.roi_merging import (
    SegmentationROI,
    merge_rois)


@pytest.fixture
def ophys_roi_list():
    roi_list = []

    fov = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           [0,0,2,0,0,0,4,0,0,0,0,0,0,0],
           [0,0,2,2,2,4,4,5,5,5,0,0,0,0],
           [0,0,2,3,3,4,4,4,8,5,0,0,0,0],
           [0,0,2,3,3,9,9,8,8,5,0,0,0,0],
           [0,0,0,3,3,7,9,8,8,0,0,0,0,0],
           [0,1,6,6,7,7,9,9,0,0,0,0,0,0],
           [0,1,6,6,7,0,0,0,0,0,0,0,0,0],
           [0,1,6,0,0,0,0,0,0,0,0,0,0,0],
           [0,1,1,0,0,0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
           [0,0,0,0,0,0,0,0,0,0,0,0,0,0]]

    fov = np.array(fov).astype(int)
    for ii in range(1,10,1):
        pixels = np.where(fov==ii)
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
    for roi in ophys_roi_list:
        f = rng.random_sample()
        s_roi = SegmentationROI.from_ophys_roi(roi, flux_value=f)
        compare_ophys_rois(roi, s_roi)
        assert np.abs(s_roi.flux_value-f) < 1.0e-10
        assert len(s_roi.ancestors) == 0

        # check that you can get self from get_ancestor
        anc = s_roi.get_ancestor(roi.roi_id)
        compare_segmentation_rois(anc, s_roi)
