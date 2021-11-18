import pytest
import numpy as np
from itertools import product

from ophys_etl.utils.array_utils import downsample_array

from ophys_etl.modules.median_filtered_max_projection.utils import (
    apply_median_filter_to_video,
    median_filtered_max_projection_from_array,
    median_filtered_max_projection_from_path)


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


@pytest.mark.parametrize('n_processors, input_frame_rate, kernel_size',
                         product((2, 3, 4),
                                 (3.0, 6.0, 11.0, 31.0),
                                 (2, 3, 4)))
def test_median_filtered_max_projection_from_array(
        n_processors,
        input_frame_rate,
        kernel_size):

    rng = np.random.default_rng(77123)
    video = rng.random((1000, 40, 47))

    downsampled_frame_rate = 4.0

    if input_frame_rate < 4.0:
        decimated_video = video
    else:
        decimated_video = downsample_array(
                             video,
                             input_fps=input_frame_rate,
                             output_fps=downsampled_frame_rate,
                             strategy='average')

    filtered_video = apply_median_filter_to_video(decimated_video,
                                                  kernel_size)

    expected = np.max(filtered_video, axis=0)
    assert expected.shape == (40, 47)

    actual = median_filtered_max_projection_from_array(
                video,
                input_frame_rate,
                downsampled_frame_rate,
                kernel_size,
                n_processors)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
        "input_frame_rate, downsampled_frame_rate, median_filter_kernel_size, "
        "n_frames_at_once",
        product((31.0, 11.0), (7.0, 4.0), (3, 4), (100, 52, -1, 0)))
def test_maximum_projection_from_path(
        video_path_fixture,
        video_data_fixture,
        input_frame_rate,
        downsampled_frame_rate,
        median_filter_kernel_size,
        n_frames_at_once):

    expected = median_filtered_max_projection_from_array(
                    video_data_fixture,
                    input_frame_rate,
                    downsampled_frame_rate,
                    median_filter_kernel_size,
                    3)

    actual = median_filtered_max_projection_from_path(
                    video_path_fixture,
                    input_frame_rate,
                    downsampled_frame_rate,
                    median_filter_kernel_size,
                    3,
                    n_frames_at_once=n_frames_at_once)

    np.testing.assert_array_equal(actual, expected)
