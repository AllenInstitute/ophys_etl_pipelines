import pytest
import numpy as np
from ophys_etl.utils import array_utils as au


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
                    [31, 95, 159],
                    [223, 255, 255]]))
                ]
        )
def test_normalize_array(array, lower_cutoff, upper_cutoff, expected):
    normalized = au.normalize_array(array, lower_cutoff, upper_cutoff)
    np.testing.assert_array_equal(normalized, expected)


def test_pairwise_distance():
    rng = np.random.default_rng(77123)
    data = rng.random((20, 72))
    distances = au.pairwise_distances(data)
    assert distances.shape == (20, 20)
    eps = 1.0e-20
    for ii in range(20):
        for jj in range(ii, 20, 1):
            expected = np.sqrt(np.sum((data[ii, :] - data[jj, :])**2))
            assert np.abs((expected-distances[ii, jj])/(eps+expected)) < 1.0e-6
            assert np.abs((expected-distances[jj, ii])/(eps+expected)) < 1.0e-6
