import pytest
import numpy as np
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.utils.roi_utils import (
    background_mask_from_roi_list)
from ophys_etl.modules.segmentation.filter.filter_utils import (
    mean_metric_from_roi,
    median_metric_from_roi,
    z_vs_background_from_roi)


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


def test_z_vs_background_from_roi():

    rng = np.random.default_rng(776233)
    img_shape = (32, 32)
    img = rng.normal(1.0, 0.1, img_shape)

    mask = [[True, True, True],
            [True, True, True],
            [True, True, True]]

    roi = OphysROI(x0=10, width=3, y0=10, height=3,
                   valid_roi=True, roi_id=0,
                   mask_matrix=mask)

    img[10:13, 10:13] = 1.3

    background = background_mask_from_roi_list([roi], img_shape)

    z_score = z_vs_background_from_roi(
                  roi,
                  img,
                  background,
                  n_desired_background=100)
    assert np.allclose(z_score, 3, rtol=0.1, atol=0.0)

    # make sure it can handle case where you ask for
    # more pixels than are available
    z_score = z_vs_background_from_roi(
                  roi,
                  img,
                  background,
                  n_desired_background=10000)
    assert np.allclose(z_score, 3, rtol=0.1, atol=0.0)

    with pytest.raises(RuntimeError, match='These must be equal'):
        z_vs_background_from_roi(
                  roi,
                  img,
                  np.ones((10, 10), dtype=bool),
                  n_desired_background=100)
