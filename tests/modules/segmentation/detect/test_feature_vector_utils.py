import pytest
import numpy as np

from ophys_etl.modules.segmentation.detect.feature_vector_utils import (
    choose_timesteps)


def test_choose_timesteps():
    rng = np.random.default_rng(16232213)
    movie_shape = (100, 32, 32)

    img_data = rng.normal(0.0, 0.1, movie_shape[1:])
    movie_data = rng.normal(0.0, 0.1, movie_shape)

    img_data[10, 15] = 1.0
    img_data[7, 12] = 0.9
    img_data[13, 11] = 0.8
    img_data[12, 17] = 0.75

    mx = movie_data.max()

    movie_data[0:15, 10, 15] = mx+1.0
    movie_data[15:20, 10, 15] = mx+0.5

    movie_data[20:35, 7, 12] = mx+1.0
    movie_data[35:40, 7, 12] = mx+0.5

    movie_data[70:85, 13, 11] = mx+1.0
    movie_data[85:90, 13, 11] = mx+0.5

    movie_data[65:80, 12, 17] = mx+1.0
    movie_data[80:85, 12, 17] = mx+0.5

    seed_pt = (10, 15)

    with pytest.raises(RuntimeError, match='These should be'):
        choose_timesteps(
                    movie_data,
                    seed_pt,
                    0.2,
                    img_data,
                    pixel_ignore=np.zeros((4, 4), dtype=bool))

    timesteps = choose_timesteps(
                    movie_data,
                    seed_pt,
                    0.15,
                    img_data)

    expected = np.concatenate([range(0, 15),
                               range(20, 35),
                               range(70, 85)])
    np.testing.assert_array_equal(expected, timesteps)

    timesteps = choose_timesteps(
                    movie_data,
                    seed_pt,
                    0.2,
                    img_data)

    expected = np.concatenate([range(0, 20),
                               range(20, 40),
                               range(70, 90)])
    np.testing.assert_array_equal(expected, timesteps)

    # mark 13, 11 as a pixel to ignore
    mask = np.zeros((32, 32), dtype=bool)
    mask[13, 11] = True

    timesteps = choose_timesteps(
                    movie_data,
                    seed_pt,
                    0.15,
                    img_data,
                    pixel_ignore=mask)

    expected = np.concatenate([range(0, 15),
                               range(20, 35),
                               range(65, 80)])

    np.testing.assert_array_equal(expected, timesteps)

    timesteps = choose_timesteps(
                    movie_data,
                    seed_pt,
                    0.2,
                    img_data,
                    pixel_ignore=mask)

    expected = np.concatenate([range(0, 20),
                               range(20, 40),
                               range(65, 85)])

    np.testing.assert_array_equal(expected, timesteps)
