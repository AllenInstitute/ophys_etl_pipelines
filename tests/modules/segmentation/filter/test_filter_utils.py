import pytest
import numpy as np
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.filter.filter_utils import (
    mean_metric_from_roi,
    median_metric_from_roi)


@pytest.fixture(scope='session')
def image_fixture():
    data = np.arange(1024, dtype=float).reshape(32, 32)
    return data


@pytest.mark.parametrize(
        "x0, y0, mask, expected",
        [(12, 15, [[True, False, True], [False, True, False]], 503.66667),
         (20, 5, [[True, True, False, False],
                  [False, True, False, False],
                  [True, True, True, True]], 222.28571)])
def test_mean_from_roi(image_fixture, x0, y0, mask, expected):
    roi = OphysROI(x0=x0, width=len(mask[0]),
                   y0=y0, height=len(mask),
                   valid_roi=True, roi_id=0,
                   mask_matrix=mask)
    actual = mean_metric_from_roi(roi, image_fixture)
    assert np.allclose(actual, expected, rtol=0.0, atol=0.0001)


@pytest.mark.parametrize(
        "x0, y0, mask, expected",
        [(12, 15, [[True, False, True], [False, True, False]], 494.0),
         (20, 5, [[True, True, False, False],
                  [False, True, False, False],
                  [True, True, True, True]], 244.0),
         (11, 9, [[True, False, True], [True, True, False]], 316.0)])
def test_median_from_roi(image_fixture, x0, y0, mask, expected):
    roi = OphysROI(x0=x0, width=len(mask[0]),
                   y0=y0, height=len(mask),
                   valid_roi=True, roi_id=0,
                   mask_matrix=mask)
    actual = median_metric_from_roi(roi, image_fixture)
    assert np.allclose(actual, expected, rtol=0.0, atol=0.0001)
