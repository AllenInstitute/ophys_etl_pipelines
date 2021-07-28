import pytest
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.utils.roi_utils import do_rois_abut

from ophys_etl.modules.segmentation.qc_utils.roi_comparison_utils import (
    get_roi_color_map)


@pytest.fixture(scope='session')
def roi_list():
    """
    List of OphysROI
    """
    output = []

    # construct an image with a bunch of connected
    # ROIs
    full_field = np.zeros((64, 64), dtype=int)
    full_field[32:49, 32:40] = 1
    full_field[20:34, 20:35] = 2
    full_field[34:38, 20:35] = 3
    full_field[32:37, 40:45] = 4
    full_field[32:37, 45:50] = 5
    full_field[10:17, 39:50] = 6
    full_field[10:17, 37:39] = 7
    full_field[5:10, 37:39] = 8
    full_field[34:46, 34:37] = 9

    for roi_id in range(1, 10, 1):
        valid = np.argwhere(full_field == roi_id)
        min_row = valid[:, 0].min()
        min_col = valid[:, 1].min()
        max_row = valid[:, 0].max()
        max_col = valid[:, 1].max()
        height = max_row-min_row+1
        width = max_col-min_col+1
        mask = np.zeros((height, width), dtype=bool)
        mask[valid[:, 0]-min_row, valid[:, 1]-min_col] = True
        roi = OphysROI(x0=int(min_col), width=int(width),
                       y0=int(min_row), height=int(height),
                       mask_matrix=mask,
                       roi_id=roi_id, valid_roi=True)
        output.append(roi)
    return output


def test_roi_coloring(roi_list):
    color_map = get_roi_color_map(roi_list)
    for roi in roi_list:
        assert roi.roi_id in color_map
    assert len(color_map) == len(roi_list)
    pairs = 0
    for ii in range(len(roi_list)):
        roi0 = roi_list[ii]
        for jj in range(ii+1, len(roi_list)):
            roi1 = roi_list[jj]
            if do_rois_abut(roi0, roi1):
                pairs += 1
                assert color_map[roi0.roi_id] != color_map[roi1.roi_id]
    assert pairs > 0
    assert len(set(color_map.values())) < len(roi_list)
