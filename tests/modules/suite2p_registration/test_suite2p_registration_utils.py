import pytest
import numpy as np
from ophys_etl.modules.suite2p_registration.utils import \
        identify_and_clip_outliers


@pytest.mark.parametrize(
        ("excess_indices", "deltas", "thresh", "expected_indices"), [
            ([], [], 10, []),
            ([123, 567], [20, 20], 10, [123, 567]),
            ([123, 567], [8, 20], 10, [567]),
            ([123, 567], [-20, -20], 10, [123, 567]),
            ([123, 567], [-8, -20], 10, [567]),
            ([123, 567, 1234, 5678], [-20, -8, 8, 20], 10, [123, 5678])
            ])
def test_identify_and_clip_outliers(excess_indices, deltas,
                                    thresh, expected_indices):
    frame_index = np.arange(10000)
    # long-range drifts
    baseline = 20.0 * np.cos(2.0 * np.pi * frame_index / 500)
    additional = np.zeros_like(baseline)
    for index, delta in zip(excess_indices, deltas):
        additional[index] += delta

    data, indices = identify_and_clip_outliers(
            baseline + additional,
            10,
            thresh)

    # check that the outliers were identified
    assert set(indices) == set(expected_indices)
    # check that the outlier values were clipped to within
    # the threshold of the underlying trend data
    # with a small delta value because the median-filtered data
    # is not quite ever the same as the baseline data
    deltas = np.abs(data[expected_indices] - baseline[expected_indices])
    small_delta = 1.0
    assert np.all(deltas < thresh + small_delta)
