import pytest
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.filter.roi_filter import (
    ROIMetricStatFilter)


@pytest.fixture(scope='session')
def img_fixture():
    data = np.arange(1024, dtype=float).reshape(32, 32)
    return data


@pytest.fixture(scope='session')
def roi_list_fixture():

    roi_list = []
    roi = OphysROI(x0=0, y0=3, height=3, width=3,
                   valid_roi=True,
                   roi_id=0,
                   mask_matrix=[[False, True, False],
                                [True, True, True],
                                [True, True, False]])
    roi_list.append(roi)
    roi = OphysROI(x0=3, y0=2, width=2, height=2,
                   valid_roi=True,
                   roi_id=1,
                   mask_matrix=[[True, False], [True, True]])
    roi_list.append(roi)
    roi = OphysROI(x0=9, y0=3, width=3, height=2,
                   valid_roi=True,
                   roi_id=2,
                   mask_matrix=[[True, True, True],
                                [False, True, True]])
    roi_list.append(roi)
    roi = OphysROI(x0=0, y0=1, width=2, height=3,
                   valid_roi=True,
                   roi_id=3,
                   mask_matrix=[[False, False], [True, True],
                                [False, True]])
    roi_list.append(roi)
    roi = OphysROI(x0=5, y0=7, width=2, height=2,
                   valid_roi=True,
                   roi_id=4,
                   mask_matrix=[[True, False], [True, True]])
    roi_list.append(roi)
    return roi_list


@pytest.mark.parametrize(
    "stat_name, min_val, max_val, expected_valid",
    [('mean', 88.6, None, set([0, 1, 2, 4])),
     ('mean', 88.6, 134.0, set([1, 2])),
     ('mean', None, 108.0, set([1, 3])),
     ('median', 88.8, None, set([0, 1, 2, 4])),
     ('median', 88.8, 130.0, set([0, 1, 2])),
     ('median', None, 130.0, set([0, 1, 2, 3]))]
)
def test_filter_on_stat(img_fixture, roi_list_fixture,
                        stat_name, min_val, max_val, expected_valid):

    this_filter = ROIMetricStatFilter(img_fixture,
                                      stat_name,
                                      min_metric=min_val,
                                      max_metric=max_val)

    result = this_filter.do_filtering(roi_list_fixture)
    valid_set = set([roi.roi_id for roi in result['valid_roi']])
    invalid_set = set([roi.roi_id for roi in result['invalid_roi']])
    assert len(valid_set.intersection(invalid_set)) == 0
    assert len(valid_set) + len(invalid_set) == len(roi_list_fixture)
    assert valid_set == expected_valid
