import pytest
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.utils.roi_utils import (
    background_mask_from_roi_list)

from ophys_etl.modules.segmentation.filter.filter_utils import (
    z_vs_background_from_roi)

from ophys_etl.modules.segmentation.filter.roi_filter import (
    ZvsBackgroundFilter)


@pytest.fixture(scope='session')
def img_shape_fixture():
    return (64, 64)


@pytest.fixture(scope='session')
def roi_list_fixture(img_shape_fixture):
    rng = np.random.default_rng(771223)
    roi_list = []
    for ii in range(10):
        x0 = rng.integers(0, img_shape_fixture[1]-5)
        width = min(img_shape_fixture[1]-x0,
                    rng.integers(5, 8))
        y0 = rng.integers(0, img_shape_fixture[0]-5)
        height = min(img_shape_fixture[0]-y0,
                     rng.integers(5, 8))
        mask = rng.integers(0, 2, (height, width)).astype(bool)
        roi = OphysROI(x0=int(x0), width=int(width),
                       y0=int(y0), height=int(height),
                       valid_roi=True, roi_id=ii,
                       mask_matrix=mask)
        roi_list.append(roi)
    return roi_list


@pytest.fixture(scope='session')
def img_fixture(roi_list_fixture, img_shape_fixture):

    rng = np.random.default_rng(542392)
    img = rng.normal(2.0, 0.2, img_shape_fixture)
    for roi in roi_list_fixture:
        value = rng.random()*4.0+2.0
        rows = roi.global_pixel_array[:, 0]
        cols = roi.global_pixel_array[:, 1]
        roi_values = rng.normal(value, 0.1, len(rows))
        img[rows, cols] = roi_values
    return img


@pytest.fixture(scope='session')
def ground_truth_fixture(roi_list_fixture,
                         img_shape_fixture,
                         img_fixture):
    bckgd_mask = background_mask_from_roi_list(
                       roi_list_fixture,
                       img_shape_fixture)

    z_score_lookup = dict()
    for roi in roi_list_fixture:
        z_score = z_vs_background_from_roi(
                      roi,
                      img_fixture,
                      bckgd_mask,
                      n_desired_background=100)
        z_score_lookup[roi.roi_id] = z_score
    return z_score_lookup


@pytest.mark.parametrize('cutoff', [0.0, 9.0, 19.0])
def test_z_vs_background_filter(
        roi_list_fixture,
        img_fixture,
        ground_truth_fixture,
        cutoff):

    valid_roi_id = set()
    invalid_roi_id = set()
    for roi in roi_list_fixture:
        if ground_truth_fixture[roi.roi_id] < cutoff:
            invalid_roi_id.add(roi.roi_id)
        else:
            valid_roi_id.add(roi.roi_id)

    this_filter = ZvsBackgroundFilter(
                     img_fixture,
                     cutoff,
                     100)
    results = this_filter.do_filtering(roi_list_fixture)

    actual_valid = set([r.roi_id for r in results['valid_roi']])
    actual_invalid = set([r.roi_id for r in results['invalid_roi']])
    assert actual_valid == valid_roi_id
    assert actual_invalid == invalid_roi_id
