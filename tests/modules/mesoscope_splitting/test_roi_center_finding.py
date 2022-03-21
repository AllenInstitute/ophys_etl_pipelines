import pytest
import numpy as np

from ophys_etl.modules.mesoscope_splitting.__main__ import (
    get_nearest_roi_center,
    get_valid_roi_centers)


def test_get_nearest_valid_roi_center():
    """
    Test that get_nearest_valid_roi_center correctly maps
    a candidate roi_center to its nearest valid_roi_center
    """
    valid_centers = [(1.1, 2.1),
                     (1.1, 2.2)]

    roi_list = [(1.1, 1.9),
                (1.1, 2.14),
                (1.1, 2.16)]
    expected = [0, 0, 1]
    for this_roi, this_expected in zip(roi_list, expected):
        assert this_expected == get_nearest_roi_center(
                                     this_roi_center=this_roi,
                                     valid_roi_centers=valid_centers)

    with pytest.raises(RuntimeError, match='Could not find nearest ROI'):
        get_nearest_roi_center(this_roi_center=roi_list[0],
                               valid_roi_centers=[])


def test_get_valid_roi_centers():
    """
    Test that get_valid_roi_centers returns the correct list of
    distinct ROI centers
    """

    class MockTimeSeriesSplitter(object):
        def __init__(self):
            manifest = []
            center_lookup = dict()
            for i_roi, zz, xx, yy in zip((1, 2, 3, 4),
                                         (5, 6, 7, 8),
                                         (1.1, 1.10001, 1.1, 3.1),
                                         (4.5, 4.5, 4.5001, 2.4)):
                pair = (i_roi, zz)
                manifest.append(pair)
                center = (xx, yy)
                center_lookup[i_roi] = center
            self.roi_z_int_manifest = manifest
            self.center_lookup = center_lookup

        def roi_center(self, i_roi: int):
            return self.center_lookup[i_roi]

    splitter = MockTimeSeriesSplitter()
    valid_centers = get_valid_roi_centers(splitter)
    assert len(valid_centers) == 2
    np.testing.assert_allclose(np.array([1.1, 4.5]),
                               np.array(valid_centers[0]))
    np.testing.assert_allclose(np.array([3.1, 2.4]),
                               np.array(valid_centers[1]))
