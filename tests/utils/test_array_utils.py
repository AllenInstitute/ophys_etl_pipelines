import pytest
import numpy as np
from itertools import product
from ophys_etl.utils import array_utils as au


@pytest.mark.parametrize(
        "input_frame_rate, downsampled_frame_rate, expected",
        [(22.0, 50.0, 1),
         (100.0, 25.0, 4),
         (100.0, 7.0, 14),
         (100.0, 8.0, 12),
         (100.0, 7.9, 13)])
def test_n_frames_from_hz(
        input_frame_rate, downsampled_frame_rate, expected):
    actual = au.n_frames_from_hz(input_frame_rate,
                                 downsampled_frame_rate)
    assert actual == expected


@pytest.mark.parametrize(
        ("array, input_fps, output_fps, random_seed, strategy, expected"),
        [
            (
                # random downsample 1D array
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 2, 0, 'random',
                np.array([2, 5])),
            (
                # random downsample ND array
                np.array([
                    [1, 3], [4, 4], [6, 8], [2, 1], [3, 2],
                    [5, 8], [11, 12]]),
                7, 2, 0, 'random',
                np.array([[2, 1], [5, 8]])),
            (
                # first downsample 1D array
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 2, 0, 'first',
                np.array([1, 3])),
            (
                # random downsample ND array
                np.array([
                    [1, 3], [4, 4], [6, 8], [2, 1], [3, 2],
                    [5, 8], [11, 12]]),
                7, 2, 0, 'first',
                np.array([[1, 3], [3, 2]])),
            (
                # last downsample 1D array
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 2, 0, 'last',
                np.array([2, 11])),
            (
                # last downsample ND array
                np.array([
                    [1, 3], [4, 4], [6, 8], [2, 1], [3, 2],
                    [5, 8], [11, 12]]),
                7, 2, 0, 'last',
                np.array([[2, 1], [11, 12]])),
            (
                # average downsample 1D array
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 2, 0, 'average',
                np.array([13/4, 19/3])),
            (
                # average downsample ND array
                np.array([
                    [1, 3], [4, 4], [6, 8], [2, 1], [3, 2],
                    [5, 8], [11, 12]]),
                7, 2, 0, 'average',
                np.array([[13/4, 4], [19/3, 22/3]])),
            (
                # average downsampel ND array with only 1 output frame
                np.array([[1, 2], [3, 4], [5, 6]]),
                10, 1, 0, 'average',
                np.array([[3.0, 4.0]])
            ),
            (
                # maximum downsample 1D array
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 2, 0, 'maximum',
                np.array([6, 11])),
            ])
def test_downsample(array, input_fps, output_fps, random_seed, strategy,
                    expected):
    array_out = au.downsample_array(
            array=array,
            input_fps=input_fps,
            output_fps=output_fps,
            strategy=strategy,
            random_seed=random_seed)
    assert np.array_equal(expected, array_out)


@pytest.mark.parametrize(
        ("array, input_fps, output_fps, random_seed, strategy, expected"),
        [
            (
                # upsampling not defined
                np.array([1, 4, 6, 2, 3, 5, 11]),
                7, 11, 0, 'maximum',
                np.array([6, 11])),
            (
                # maximum downsample ND array
                # not defined
                np.array([
                    [1, 3], [4, 4], [6, 8], [2, 1], [3, 2],
                    [5, 1234], [11, 12]]),
                7, 2, 0, 'maximum',
                np.array([[6, 8], [11, 12]])),
            ])
def test_downsample_exceptions(array, input_fps, output_fps, random_seed,
                               strategy, expected):
    with pytest.raises(ValueError):
        au.downsample_array(
                array=array,
                input_fps=input_fps,
                output_fps=output_fps,
                strategy=strategy,
                random_seed=random_seed)


@pytest.mark.parametrize('input_fps', [3, 4, 5])
def test_decimate_video(input_fps):
    """
    This is another test of downsample array to make sure that
    it treats video-like arrays the way our median_filtered_max_projection
    code expects
    """
    rng = np.random.default_rng(62134)
    video = rng.random((71, 40, 40))

    expected = []
    for i0 in range(0, 71, input_fps):
        frame = np.mean(video[i0:i0+input_fps, :, :],
                        axis=0)
        expected.append(frame)
    expected = np.array(expected)

    actual = au.downsample_array(video,
                                 input_fps=input_fps,
                                 output_fps=1,
                                 strategy='average')
    np.testing.assert_array_equal(expected, actual)


@pytest.mark.parametrize(
        "array, lower_cutoff, upper_cutoff, expected",
        [
            (
                np.array([
                    [0.0, 100.0, 200.0],
                    [300.0, 400.0, 500.0],
                    [600.0, 700.0, 800.0]]),
                250, 650,
                np.uint8([
                    [0, 0, 0],
                    [32, 96, 159],
                    [223, 255, 255]]))
                ]
        )
def test_normalize_array(array, lower_cutoff, upper_cutoff, expected):
    normalized = au.normalize_array(array,
                                    lower_cutoff=lower_cutoff,
                                    upper_cutoff=upper_cutoff)
    np.testing.assert_array_equal(normalized, expected)


@pytest.mark.parametrize(
        "input_array, expected_array",
        [(np.array([0, 1, 2, 3, 4, 5]).astype(int),
          np.array([0, 51, 102, 153, 204, 255]).astype(np.uint8)),
         (np.array([-1, 0, 1, 2, 4]).astype(int),
          np.array([0, 51, 102, 153, 255]).astype(np.uint8)),
         (np.array([-1.0, 1.5, 2, 3, 4]).astype(float),
          np.array([0, 128, 153, 204, 255]).astype(np.uint8))])
def test_scale_to_uint8(input_array, expected_array):
    """
    Test normalize_array when cutoffs are not specified
    """
    actual = au.normalize_array(input_array)
    np.testing.assert_array_equal(actual, expected_array)


@pytest.mark.parametrize(
        "lower_cutoff, upper_cutoff",
        product((None, 15.0), (None, 77.0)))
def test_array_to_rgb(
        lower_cutoff, upper_cutoff):

    img = np.arange(144, dtype=float).reshape(12, 12)
    scaled = au.normalize_array(array=img,
                                lower_cutoff=lower_cutoff,
                                upper_cutoff=upper_cutoff)

    rgb = au.array_to_rgb(
                input_array=img,
                lower_cutoff=lower_cutoff,
                upper_cutoff=upper_cutoff)

    assert rgb.dtype == np.uint8
    assert rgb.shape == (12, 12, 3)
    for ic in range(3):
        np.testing.assert_array_equal(rgb[:, :, ic], scaled)
