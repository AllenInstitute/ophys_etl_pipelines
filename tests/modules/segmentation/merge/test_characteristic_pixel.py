import pytest
import numpy as np
from itertools import combinations

from ophys_etl.modules.segmentation.merge.characteristic_pixel import (
    _update_key_pixel_lookup,
    _update_key_pixel_lookup_per_pix,
    update_key_pixel_lookup)

from ophys_etl.modules.segmentation.merge.roi_time_correlation import (
    get_characteristic_timeseries)


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

    assert len(result) == len(roi_id_list)

    for roi_id in roi_id_list:
        expected = get_characteristic_timeseries(
                                       small_rois[roi_id],
                                       filter_fraction=filter_fraction)
        np.testing.assert_allclose(expected,
                                   result[roi_id],
                                   atol=1.0e-10,
                                   rtol=1.0e-10)


@pytest.mark.parametrize('filter_fraction, n_processors',
                         [(0.2, 2), (0.2, 3),
                          (0.3, 2), (0.3, 3)])
def test_update_key_pixel(small_rois,
                          filter_fraction,
                          n_processors):
    roi_id_list = list(small_rois.keys())
    result = _update_key_pixel_lookup(
                     roi_id_list,
                     small_rois,
                     filter_fraction,
                     n_processors)

    assert len(result) == len(roi_id_list)

    for roi_id in roi_id_list:
        expected = get_characteristic_timeseries(
                                       small_rois[roi_id],
                                       filter_fraction=filter_fraction)
        np.testing.assert_allclose(expected,
                                   result[roi_id],
                                   atol=1.0e-10,
                                   rtol=1.0e-10)


@pytest.mark.parametrize('filter_fraction, n_processors',
                         [(0.2, 2), (0.2, 3),
                          (0.3, 2), (0.3, 3)])
def test_full_update_key_pixel(small_rois,
                               large_rois,
                               filter_fraction,
                               n_processors):
    video_lookup = {}
    video_lookup.update(small_rois)
    video_lookup.update(large_rois)

    merger_list = list(combinations(range(8), 2))
    result = update_key_pixel_lookup(
                   merger_list,
                   {},
                   video_lookup,
                   filter_fraction,
                   n_processors,
                   size_threshold=30)

    assert len(result) == (len(small_rois) + len(large_rois))
    for roi_id in small_rois:
        assert result[roi_id]['area'] == small_rois[roi_id].shape[1]
        expected = get_characteristic_timeseries(
                                       small_rois[roi_id],
                                       filter_fraction=filter_fraction)
        np.testing.assert_allclose(expected,
                                   result[roi_id]['key_pixel'],
                                   atol=1.0e-10,
                                   rtol=1.0e-10)

    for roi_id in large_rois:
        assert result[roi_id]['area'] == large_rois[roi_id].shape[1]
        expected = get_characteristic_timeseries(
                       large_rois[roi_id],
                       filter_fraction=filter_fraction)
        np.testing.assert_allclose(expected,
                                   result[roi_id]['key_pixel'],
                                   atol=1.0e-10,
                                   rtol=1.0e-10)
