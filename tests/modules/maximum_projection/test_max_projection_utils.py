import pytest
import numpy as np
from itertools import product

from ophys_etl.modules.maximum_projection.utils import (
    apply_median_filter_to_video,
    decimate_video,
    generate_max_projection,
    scale_to_uint8,
    n_frames_from_hz)


@pytest.mark.parametrize(
        "input_array, expected_array",
        [(np.array([0, 1, 2, 3, 4, 5]).astype(int),
          np.array([0, 51, 102, 153, 204, 255]).astype(np.uint8)),
         (np.array([-1, 0, 1, 2, 4]).astype(int),
          np.array([0, 51, 102, 153, 255]).astype(np.uint8)),
         (np.array([-1.0, 1.5, 2, 3, 4]).astype(float),
          np.array([0, 128, 153, 204, 255]).astype(np.uint8))])
def test_scale_to_uint8(input_array, expected_array):
    actual = scale_to_uint8(input_array)
    np.testing.assert_array_equal(actual, expected_array)


@pytest.mark.parametrize(
        "input_frame_rate, downsampled_frame_rate, expected",
        [(22.0, 50.0, 1),
         (100.0, 25.0, 4),
         (100.0, 7.0, 14),
         (100.0, 8.0, 12),
         (100.0, 7.9, 13)])
def test_n_frames_from_hz(
        input_frame_rate, downsampled_frame_rate, expected):
    actual = n_frames_from_hz(input_frame_rate, downsampled_frame_rate)
    assert actual == expected


@pytest.mark.parametrize('kernel_size', [3, 5, 7])
def test_median_filter_to_video(kernel_size):
    # test will fail on even number kernel sizes;
    # not entirely sure how ndimage handles kernels that don't
    # have a center pixel

    rng = np.random.default_rng(17231)
    chunk_of_frames = rng.random((10, 20, 20))
    assert chunk_of_frames.shape == (10, 20, 20)

    kernel_1d = np.arange(-np.floor(kernel_size//2),
                          kernel_size-np.floor(kernel_size//2),
                          1,
                          dtype=int)

    assert len(kernel_1d) == kernel_size

    expected = np.zeros(chunk_of_frames.shape, dtype=float)
    for i_frame in range(chunk_of_frames.shape[0]):
        for irow in range(chunk_of_frames.shape[1]):
            for icol in range(chunk_of_frames.shape[2]):
                neighbors = np.zeros(kernel_size**2, dtype=float)
                for ineigh, (drow, dcol) in enumerate(product(kernel_1d,
                                                              kernel_1d)):
                    r = irow+drow
                    if r < 0:
                        r = -1*r - 1
                    elif r >= chunk_of_frames.shape[1]:
                        r = 2*(chunk_of_frames.shape[1]-1) - r + 1

                    c = icol+dcol
                    if c < 0:
                        c = -1*c - 1
                    elif c >= chunk_of_frames.shape[2]:
                        c = 2*(chunk_of_frames.shape[2]-1) - c + 1

                    neighbors[ineigh] = chunk_of_frames[i_frame, r, c]
                expected[i_frame, irow, icol] = np.median(neighbors)

    actual = apply_median_filter_to_video(chunk_of_frames, kernel_size)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('frames_to_group', [3, 4, 5])
def test_decimate_video(frames_to_group):
    rng = np.random.default_rng(62134)
    video = rng.random((71, 40, 40))

    expected = []
    for i0 in range(0, 71, frames_to_group):
        frame = np.mean(video[i0:i0+frames_to_group, :, :],
                        axis=0)
        expected.append(frame)
    expected = np.array(expected)

    actual = decimate_video(video, frames_to_group)
    np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize('n_processors, input_frame_rate, kernel_size',
                         product((2, 3, 4),
                                 (3.0, 6.0, 11.0, 31.0),
                                 (2, 3, 4)))
def test_generate_max_projection(
        n_processors,
        input_frame_rate,
        kernel_size):

    rng = np.random.default_rng(77123)
    video = rng.random((1000, 40, 47))

    downsampled_frame_rate = 4.0

    if input_frame_rate < 4.0:
        frames_to_group = 1
        decimated_video = video
    else:
        frames_to_group = np.round(input_frame_rate/downsampled_frame_rate)
        frames_to_group = frames_to_group.astype(int)
        decimated_video = decimate_video(video, frames_to_group)

    filtered_video = apply_median_filter_to_video(decimated_video,
                                                  kernel_size)

    expected = np.max(filtered_video, axis=0)
    assert expected.shape == (40, 47)

    actual = generate_max_projection(
                video,
                input_frame_rate,
                downsampled_frame_rate,
                kernel_size,
                n_processors)

    np.testing.assert_array_equal(actual, expected)
