import pytest
import numpy as np

from ophys_etl.modules.segmentation.merge.characteristic_pixel import (
    _update_key_pixel_lookup,
    _update_key_pixel_lookup_per_pix,
    update_key_pixel_lookup)

from ophys_etl.modules.segmentation.merge.roi_time_correlation import (
    get_brightest_pixel)


@pytest.fixture
def small_rois():
    video_lookup = {}
    rng = np.random.RandomState(881223)
    for ii in range(5):
        area = rng.randint(9,22)
        video_lookup[ii] = rng.random_sample((100, area))
    return video_lookup

@pytest.fixture
def large_rois():
    video_lookup = {}
    rng = np.random.RandomState(881223)
    for ii in range(5,8,1):
        area = rng.randint(509,622)
        video_lookup[ii] = rng.random_sample((100, area))
    return video_lookup


@pytest.mark.parametrize('filter_fraction, n_processors',
                          [(0.2, 2), (0.2, 3),
                           (0.3, 2), (0.3, 3)])
def test_update_key_pixel_per_pix(small_rois,
                                  filter_fraction,
                                  n_processors):
    roi_id_list = list(small_rois.keys())
    result = _update_key_pixel_lookup_per_pix(
                     roi_id_list,
                     small_rois,
                     filter_fraction,
                     n_processors)

    for roi_id in roi_id_list:
        expected = get_brightest_pixel(small_rois[roi_id],
                                       filter_fraction=filter_fraction)
        np.testing.assert_allclose(expected,
                                   result[roi_id],
                                   atol=1.0e-10,
                                   rtol=1.0e-10)
