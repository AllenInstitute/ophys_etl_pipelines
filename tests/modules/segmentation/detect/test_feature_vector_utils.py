import pytest
import numpy as np
from itertools import product

from ophys_etl.modules.segmentation.detect.feature_vector_utils import (
    choose_extreme_pixels,
    choose_timesteps,
    select_window_size)


@pytest.mark.parametrize(
    "ignored_pixels, true_std",
    [(None, 36.6949),
     (np.arange(10, 35), 27.4285),
     (np.arange(22), 28.54051),
     (np.arange(88, 100), 32.2470644),
     (np.concatenate([np.arange(15),
                      np.arange(44, 56),
                      np.arange(80, 100)]),
      28.16985)]
)
def test_choose_extreme_pixels(ignored_pixels, true_std):
    flat_image = np.arange(100)
    image_data = flat_image.reshape((10, 10))

    with pytest.raises(RuntimeError, match="should be the same"):
        pixel_mask = np.zeros((4, 4), dtype=bool)
        choose_extreme_pixels(image_data, [1.0], pixel_ignore=pixel_mask)

    pixel_ignore = None
    if ignored_pixels is not None:
        flat_mask = np.zeros(100, dtype=bool)
        flat_mask[ignored_pixels] = True
        pixel_ignore = flat_mask.reshape((10, 10))

    for quantiles in ([0.1], [0.1, 0.2], [0.1, 0.7, 0.9]):

        masked_image = np.copy(flat_image)
        full_image = np.copy(flat_image)
        if pixel_ignore is not None:
            masked_image = masked_image[np.logical_not(flat_mask)]
            full_image[flat_mask] = 99999.0
        expected_points = []
        for q in quantiles:
            f = np.quantile(masked_image, q)
            ii = np.argmin(np.abs(full_image-f))
            expected_points.append(tuple(np.unravel_index(ii,
                                                          image_data.shape)))

        chosen_points = choose_extreme_pixels(
                              image_data,
                              quantiles,
                              pixel_ignore=pixel_ignore)

        # check that flux values are in sorted order
        flux_values = [image_data[p[0], p[1]] for p in chosen_points]
        for ii in range(1, len(flux_values), 1):
            assert flux_values[ii] >= flux_values[ii-1]

        for expected in expected_points:
            assert expected in chosen_points


@pytest.mark.skip('broken by change')
def test_choose_timesteps():
    rng = np.random.default_rng(16232213)
    movie_shape = (100, 32, 32)

    image_data = rng.normal(0.0, 0.1, movie_shape[1:])
    movie_data = rng.normal(0.0, 0.1, movie_shape)

    image_data[10, 15] = 1.0
    image_data[7, 12] = 0.9
    image_data[13, 11] = 0.8
    image_data[12, 17] = 0.75

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
                    image_data,
                    pixel_ignore=np.zeros((4, 4), dtype=bool))

    timesteps = choose_timesteps(
                    movie_data,
                    seed_pt,
                    0.15,
                    image_data)

    chosen_pixels = choose_extreme_pixels(
                       image_data, [0.1, 0.9])

    expected_timesteps = []
    chosen_pixels.append(seed_pt)
    for p in chosen_pixels:
        trace = movie_data[:, p[0], p[1]]
        th = np.quantile(trace, 0.85)
        expected_timesteps.append(np.where(trace >= th)[0])
    expected_timesteps = np.unique(np.concatenate(expected_timesteps))
    np.testing.assert_array_equal(timesteps, expected_timesteps)

    timesteps = choose_timesteps(
                    movie_data,
                    seed_pt,
                    0.2,
                    image_data)

    chosen_pixels = choose_extreme_pixels(
                       image_data, [0.1, 0.9])

    expected_timesteps = []
    chosen_pixels.append(seed_pt)
    for p in chosen_pixels:
        trace = movie_data[:, p[0], p[1]]
        th = np.quantile(trace, 0.8)
        expected_timesteps.append(np.where(trace >= th)[0])
    expected_timesteps = np.unique(np.concatenate(expected_timesteps))
    np.testing.assert_array_equal(timesteps, expected_timesteps)

    # mark 13, 11 as a pixel to ignore
    mask = np.zeros((32, 32), dtype=bool)
    mask[13, 11] = True

    timesteps = choose_timesteps(
                    movie_data,
                    seed_pt,
                    0.15,
                    image_data,
                    pixel_ignore=mask)

    chosen_pixels = choose_extreme_pixels(
                       image_data, [0.1, 0.9],
                       pixel_ignore=mask)

    expected_timesteps = []
    chosen_pixels.append(seed_pt)
    for p in chosen_pixels:
        trace = movie_data[:, p[0], p[1]]
        th = np.quantile(trace, 0.85)
        expected_timesteps.append(np.where(trace >= th)[0])
    expected_timesteps = np.unique(np.concatenate(expected_timesteps))
    np.testing.assert_array_equal(timesteps, expected_timesteps)

    timesteps = choose_timesteps(
                    movie_data,
                    seed_pt,
                    0.2,
                    image_data,
                    pixel_ignore=mask)

    chosen_pixels = choose_extreme_pixels(
                       image_data, [0.1, 0.9],
                       pixel_ignore=mask)

    expected_timesteps = []
    chosen_pixels.append(seed_pt)
    for p in chosen_pixels:
        trace = movie_data[:, p[0], p[1]]
        th = np.quantile(trace, 0.8)
        expected_timesteps.append(np.where(trace >= th)[0])
    expected_timesteps = np.unique(np.concatenate(expected_timesteps))
    np.testing.assert_array_equal(timesteps, expected_timesteps)


@pytest.fixture(scope='session')
def image_fixture():

    rng = np.random.default_rng(76143232)
    image_data = rng.normal(0.0, 0.1, (32, 32))
    return image_data


@pytest.fixture(scope='session')
def mask_fixture():
    rng = np.random.default_rng(554433)
    return rng.integers(0, 2, (32, 32)).astype(bool)


@pytest.mark.parametrize(
        'seed_pt',
        [(10, 11), (0, 3), (3, 0), (31, 5), (6, 31)])
def test_select_window_size(image_fixture, mask_fixture, seed_pt):

    window_min = 5

    # put a "cell" in the image
    image_data = np.copy(image_fixture)

    rows, cols = np.meshgrid(range(32), range(32), indexing='ij')
    dsq = (rows-seed_pt[0])**2+(cols-seed_pt[1])**2
    valid = dsq < 36.0

    image_data[valid] += 1.0

    with pytest.raises(RuntimeError, match='These should be the same'):
        select_window_size(
                    seed_pt,
                    image_data,
                    2.0,
                    window_min=window_min,
                    window_max=1000,
                    pixel_ignore=np.zeros((2, 2), dtype=bool))

    found_z = 0
    for z_score, mask in product((1.0, 6.2, 8.4, 2.0e6),
                                 (None, mask_fixture)):
        window = select_window_size(
                    seed_pt,
                    image_data,
                    z_score,
                    window_min=window_min,
                    window_max=1000,
                    pixel_ignore=mask)

        # calculate the z-score associated with this window
        r0 = max(0, seed_pt[0]-window)
        r1 = min(image_data.shape[0], seed_pt[0]+window+1)
        c0 = max(0, seed_pt[1]-window)
        c1 = min(image_data.shape[1], seed_pt[1]+window+1)
        local_image = image_data[r0:r1, c0:c1].flatten()
        if mask is not None:
            local_image = local_image[
                            np.logical_not(mask[r0:r1, c0:c1]).flatten()]

        mu = np.mean(local_image)
        q25, q75 = np.quantile(local_image, (0.25, 0.75))
        std = (q75-q25)/1.34896
        z_actual = (image_data[seed_pt[0], seed_pt[1]]-mu)/std

        if z_actual < z_score:
            # must be the whole field of view
            assert r0 == 0
            assert r1 == image_data.shape[0]
            assert c0 == 0
            assert c1 == image_data.shape[1]
        else:
            if window != window_min:
                # calculate the z-score associated with the previous window
                w = window_min
                while w < window:
                    window_less = w
                    w = 3*w//2
                r0 = max(0, seed_pt[0]-window_less)
                r1 = min(image_data.shape[0], seed_pt[0]+window_less+1)
                c0 = max(0, seed_pt[1]-window_less)
                c1 = min(image_data.shape[1], seed_pt[1]+window_less+1)
                local_image = image_data[r0:r1, c0:c1].flatten()
                if mask is not None:
                    local_image = local_image[
                                    np.logical_not(mask[r0:r1,
                                                        c0:c1]).flatten()]

                mu = np.mean(local_image)
                q25, q75 = np.quantile(local_image, (0.25, 0.75))
                std = (q75-q25)/1.34896
                z_less = (image_data[seed_pt[0], seed_pt[1]]-mu)/std
                assert z_less < z_score
                found_z += 1

    assert found_z > 0
