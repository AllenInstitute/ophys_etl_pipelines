import h5py
import pytest
import numpy as np
from pathlib import Path
from ophys_etl.modules.suite2p_registration.utils import \
        identify_and_clip_outliers, check_movie_against_raw


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


def test_check_movie_against_raw():
    """Test that we can recognize an unordered or non flux conserving motion
    corrected movie as expected.
    """
    rng = np.random.default_rng(1234)

    file_loc = Path(__file__)
    resource_loc = file_loc.parent / 'resources'
    h5_file_loc = resource_loc / '792757260_test_data.h5'
    h5_key = 'input_frames'

    with h5py.File(h5_file_loc) as data_file:
        corr_data = data_file[h5_key][:]

    #  This should pass without a raise.
    check_movie_against_raw(corr_data=corr_data,
                            raw_hdf5=h5_file_loc,
                            h5py_key=h5_key)

    # Test that the check fails if we put in a movie of different length
    # than the raw.
    rand_data = rng.integers(0,
                             corr_data.shape[0],
                             size=(corr_data.shape[0] + 1,
                                   corr_data.shape[1],
                                   corr_data.shape[2]))
    with pytest.raises(ValueError, match=r'Length of motion .*'):
        check_movie_against_raw(corr_data=rand_data,
                                raw_hdf5=h5_file_loc,
                                h5py_key=h5_key)

    # Test that the code finds shuffled frames.
    rand_index = rng.integers(0, corr_data.shape[0], size=corr_data.shape[0])
    with pytest.raises(ValueError, match=r"The distribution of pixel .*"):
        check_movie_against_raw(corr_data=corr_data[rand_index],
                                raw_hdf5=h5_file_loc,
                                h5py_key=h5_key)
