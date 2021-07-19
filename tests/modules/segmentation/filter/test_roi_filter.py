import pytest
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysROI)

from ophys_etl.modules.segmentation.filter.roi_filter import (
    ROIBaseFilter,
    ROIAreaFilter)


@pytest.fixture
def roi_dict():

    rng = np.random.default_rng(66523)
    roi_dict = dict()

    for roi_id in range(1, 7, 1):
        x0 = rng.integers(0, 100)
        y0 = rng.integers(0, 100)
        width = rng.integers(10, 11)
        height = rng.integers(10, 15)
        mask = np.zeros(width*height).astype(bool)
        dexes = np.arange(len(mask), dtype=int)
        chosen = rng.choice(dexes, size=2*roi_id, replace=False)
        mask[chosen] = True
        roi = OphysROI(x0=int(x0), width=int(width),
                       y0=int(y0), height=int(height),
                       mask_matrix=mask.reshape((height, width)),
                       roi_id=roi_id,
                       valid_roi=True)
        assert roi.area == 2*roi_id
        roi_dict[roi_id] = roi
    return roi_dict


def test_base_roi_filter():

    class DummyFilter(ROIBaseFilter):

        def is_roi_valid(self, roi):
            return {'valid_roi': [],
                    'invalid_roi': []}

    roi_filter = DummyFilter()

    with pytest.raises(NotImplementedError,
                       match="self._reason not defined"):
        roi_filter.reason


def test_invalid_area_roi_filter(roi_dict):
    with pytest.raises(RuntimeError, match='Both max_area and min_area'):
        ROIAreaFilter()


@pytest.mark.parametrize(
        'min_area, max_area, expected_valid',
        [(None, 8, set([1, 2, 3, 4])),
         (6, None, set([3, 4, 5, 6])),
         (4, 8, set([2, 3, 4]))
         ])
def test_area_roi_filter(roi_dict, min_area, max_area, expected_valid):

    area_filter = ROIAreaFilter(min_area=min_area,
                                max_area=max_area)

    assert area_filter.reason == 'area'.encode('utf-8')

    results = area_filter.do_filtering(list(roi_dict.values()))

    valid_lookup = {roi.roi_id: roi for roi in results['valid_roi']}
    invalid_lookup = {roi.roi_id: roi for roi in results['invalid_roi']}
    assert (len(results['valid_roi'])
            + len(results['invalid_roi'])) == len(roi_dict)

    for roi_id in roi_dict:
        if roi_id in expected_valid:
            assert roi_id in valid_lookup
            actual_roi = valid_lookup[roi_id]
            assert actual_roi.valid_roi
        else:
            assert roi_id in invalid_lookup
            actual_roi = invalid_lookup[roi_id]
            assert not actual_roi.valid_roi

        expected_roi = roi_dict[roi_id]
        assert expected_roi.x0 == actual_roi.x0
        assert expected_roi.y0 == actual_roi.y0
        assert expected_roi.width == actual_roi.width
        assert expected_roi.height == actual_roi.height
        np.testing.assert_array_equal(expected_roi.mask_matrix,
                                      actual_roi.mask_matrix)
