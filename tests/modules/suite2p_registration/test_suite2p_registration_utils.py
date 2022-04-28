import h5py
from itertools import product
import pytest
import numpy as np
from pathlib import Path
import tempfile
import warnings

from ophys_etl.modules.suite2p_registration.utils import (
    identify_and_clip_outliers, check_movie_against_raw, reset_frame_shift,
    find_movie_start_end_empty_frames, check_and_warn_on_datatype
)


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


@pytest.mark.parametrize(
    ("start_frames", "end_frames", "mid_frames_low", "mid_frames_high"),
    product([0, 1], [0, 1], [False, True], [False, True]))
def test_find_movie_start_end_empty_frames(start_frames,
                                           end_frames,
                                           mid_frames_low,
                                           mid_frames_high):
    rng = np.random.default_rng(1234)
    n_frames = 100
    xy_size = 100
    tmp_hdf5_file = tempfile.mkstemp()
    frames = rng.integers(90, 110, size=(n_frames, xy_size, xy_size))
    for idx in range(start_frames):
        frames[idx] = frames[idx] / 100
    for idx in range(n_frames - end_frames, n_frames):
        frames[idx] = frames[idx] / 100
    if mid_frames_low:
        idx = n_frames // 2 - 1
        frames[idx] = frames[idx] / 100
    if mid_frames_high:
        idx = n_frames // 2 + 1
        frames[idx] = frames[idx] / 100
    with h5py.File(tmp_hdf5_file[1], 'w') as h5_file:
        h5_file.create_dataset(name='data', data=frames, chunks=frames.shape)

    lowside, highside = find_movie_start_end_empty_frames(
        h5py_name=tmp_hdf5_file[1],
        h5py_key='data',
        logger=print)

    if mid_frames_low:
        expected_low = 0
    else:
        expected_low = start_frames
    if mid_frames_high:
        expected_high = 0
    else:
        expected_high = end_frames

    assert lowside == expected_low
    assert highside == expected_high


def test_reset_frame_shift():
    """Test that frames are correctly shifted to their original location and
    shifts are reset to zero.
    """
    n_frames = 100
    movie_shape = (n_frames, 5, 5)
    bad_frame_low_idx = 0
    bad_frame_high_idx = -1
    bad_frame_start = 1
    bad_frame_end = 1

    # Create empty initial data.
    shifts_y = np.ones(n_frames, dtype=int)
    shifts_x = np.ones(n_frames, dtype=int)
    frames = np.zeros(movie_shape, dtype=int)

    # Create a dataset with the start and end frames slightly shifted
    # relative to the rest of the "movie".
    frames[bad_frame_start:n_frames - bad_frame_end, 2, 2] = 100
    frames[:bad_frame_start, 1, 1] = 100
    frames[n_frames - bad_frame_end:, 1, 1] = 100

    # Our expected movie is a set of identical frames with one "hot" pixel
    # in the center.
    expected_frames = np.zeros(movie_shape)
    expected_frames[:, 2, 2] = 100

    reset_frame_shift(
        frames, shifts_y, shifts_x, bad_frame_start, bad_frame_end)

    # Test that the frames have been shifted as expected.
    np.testing.assert_array_equal(frames, expected_frames)
    # Test that the shifts have been reset.
    assert shifts_y[bad_frame_low_idx] == 0
    assert shifts_x[bad_frame_low_idx] == 0
    assert shifts_y[bad_frame_high_idx] == 0
    assert shifts_x[bad_frame_high_idx] == 0


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


def test_check_and_warn_on_datatype():
    """Test that warnings are thrown when the wrong types data types are
    specified.
    """
    h5_file = tempfile.NamedTemporaryFile('w', suffix='.h5')

    # test no warnings
    with h5py.File(h5_file.name, 'w') as h5:
        h5.create_dataset(name='data',
                          data=np.arange(20, dtype='int16'))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_and_warn_on_datatype(h5_file.name, 'data', warnings.warn)

    # test wrong type.
    with h5py.File(h5_file.name, 'w') as h5:
        h5.create_dataset(name='data',
                          data=np.arange(20, dtype='uint16'))
    with pytest.warns(UserWarning, match='Data type is'):
        check_and_warn_on_datatype(h5_file.name, 'data', warnings.warn)

    # test wrong endian.
    with h5py.File(h5_file.name, 'w') as h5:
        h5.create_dataset(name='data',
                          data=np.arange(20, dtype='>i2'))
    with pytest.warns(UserWarning, match='Data byteorder is'):
        check_and_warn_on_datatype(h5_file.name, 'data', warnings.warn)

    h5_file.close()
