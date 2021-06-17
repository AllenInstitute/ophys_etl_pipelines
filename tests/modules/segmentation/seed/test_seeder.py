import pytest
import h5py
import numpy as np

from ophys_etl.modules.segmentation.seed.utils import Seed
from ophys_etl.modules.segmentation.seed import seeder


def test_SeederBase_init():
    fov_shape = (512, 512)
    exclusion_buffer = 3
    sb = seeder.SeederBase(fov_shape=fov_shape,
                           exclusion_buffer=exclusion_buffer)
    assert sb._fov_shape == fov_shape
    assert sb._exclusion_buffer == exclusion_buffer
    for attr in ["_candidate_seeds", "_provided_seeds",
                 "_excluded_seeds", "_excluded_pixels"]:
        assert hasattr(sb, attr)


def test_SeederBase_exclude(monkeypatch):
    def dummy_dilate(pixels, dummay_arg1, dummay_arg2):
        return pixels
    monkeypatch.setattr(seeder, "dilated_coordinates", dummy_dilate)
    pixels = {(1, 1), (3, 3), (5, 8)}
    sb = seeder.SeederBase(fov_shape=(512, 512))
    sb.exclude_pixels(pixels)
    assert sb._excluded_pixels == pixels


@pytest.mark.parametrize(
        "seeds, exclude_by_roi, expected",
        [
            (
                [
                    Seed(coordinates=(0, 0)),
                    Seed(coordinates=(0, 1)),
                    Seed(coordinates=(0, 2))],
                [False, False, False],
                [(0, 0), (0, 1), (0, 2)]),
            (
                [
                    Seed(coordinates=(0, 0)),
                    Seed(coordinates=(0, 1)),
                    Seed(coordinates=(0, 2))],
                [False, True, False],
                [(0, 0), (0, 2)]),
            ])
def test_SeederBase_iter(seeds, exclude_by_roi, expected):
    sb = seeder.SeederBase(fov_shape=(512, 512))
    sb._candidate_seeds = list(seeds)
    # manually add to excluded_pixels list
    for seed, exclude in zip(seeds, exclude_by_roi):
        if exclude:
            sb._excluded_pixels.add(seed['coordinates'])
    for pixel, seed in zip(expected, sb):
        assert pixel == seed

    # check that the iteration stops by itself (not zipped with expected)
    sb = seeder.SeederBase(fov_shape=(512, 512))
    sb._candidate_seeds = list(seeds)
    # manually add to excluded_pixels list
    for seed, exclude in zip(seeds, exclude_by_roi):
        if exclude:
            sb._excluded_pixels.add(seed['coordinates'])
    assert len(list(sb)) == len(expected)


def test_SeederBase_no_select():
    sb = seeder.SeederBase(fov_shape=(512, 512))
    with pytest.raises(NotImplementedError):
        sb.select_seeds()


def test_ImageBlockMetricSeeder_init():
    fov_shape = (512, 512)
    exclusion_buffer = 3
    seeder_grid_size = 4
    keep_fraction = 0.5
    sb = seeder.ImageBlockMetricSeeder(fov_shape=fov_shape,
                                       exclusion_buffer=exclusion_buffer,
                                       keep_fraction=keep_fraction,
                                       seeder_grid_size=seeder_grid_size)
    assert sb._fov_shape == fov_shape
    assert sb._exclusion_buffer == exclusion_buffer
    assert sb._seeder_grid_size == seeder_grid_size
    assert sb._keep_fraction == keep_fraction
    for attr in ["_candidate_seeds", "_provided_seeds", "_excluded_seeds",
                 "_excluded_pixels", "_seed_image"]:
        assert hasattr(sb, attr)


@pytest.mark.parametrize(
        "image, sigma, percentage, expected",
        [
            (
                # no smoothing
                np.array([[0.0, 0.0, 0.4, 0.0],
                          [0.5, 0.0, 0.3, 0.0],
                          [0.0, 0.5, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0]]),
                None,
                0.25,
                [(1, 0), (2, 1), (0, 2), (1, 2)]),
            (
                # a little smoothing
                np.array([[0.0, 0.0, 0.4, 0.0],
                          [0.5, 0.0, 0.3, 0.0],
                          [0.0, 0.5, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0]]),
                1,
                0.25,
                [(1, 1), (2, 1), (1, 2), (1, 0)]),
            (
                # no smoothing, keep a little more by precentage
                np.array([[0.0, 0.0, 0.4, 0.0],
                          [0.5, 0.0, 0.3, 0.0],
                          [0.0, 0.5, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 0.0]]),
                None,
                6 / 16,
                [(1, 0), (2, 1), (0, 2), (1, 2), (0, 0), (0, 1)]),
            ])
def test_ImageBlockMetricSeeder_select(image, sigma, percentage, expected):
    sb = seeder.ImageBlockMetricSeeder(fov_shape=(12, 12),
                                       keep_fraction=percentage,
                                       seeder_grid_size=1)
    sb.select_seeds(image, sigma)
    found_seeds = [i['coordinates']
                   for i in sb._candidate_seeds]
    assert found_seeds == expected


def test_ImageBlockMetricSeeder_log(tmpdir):
    image = np.array([[0.0, 0.0, 0.4, 0.0],
                      [0.5, 0.0, 0.3, 0.0],
                      [0.0, 0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
    sb = seeder.ImageBlockMetricSeeder(keep_fraction=1.0,
                                       seeder_grid_size=1)
    sb.select_seeds(image, sigma=None)

    h5path = tmpdir / "seed_qc.h5"
    with h5py.File(h5path, "w") as f:
        sb.log_to_h5_group(f)

    with h5py.File(h5path, "r") as f:
        assert "seeding" in f
        group = f["seeding"]
        for k in ["provided_seeds", "excluded_seeds",
                  "exclusion_reason", "seed_image"]:
            assert k in group


@pytest.mark.parametrize(
        "image, n_samples, minimum_distance, excluded, expected",
        [
            (
                np.array([[0, 1, 2, 3],
                          [4, 5, 6, 7],
                          [8, 9, 10, 11],
                          [12, 13, 14, 15]]),
                3, 2, [],
                [[(3, 3), (3, 1), (1, 3)],
                 [(3, 2), (3, 0), (1, 2)],
                 [(2, 3), (2, 1), (0, 3)],
                 [(2, 2), (2, 0), (0, 2)],
                 [(1, 1)],
                 [(1, 0)],
                 [(0, 1)],
                 [(0, 0)]]),
            (
                np.array([[0, 1, 2, 3],
                          [4, 5, 6, 7],
                          [8, 9, 10, 11],
                          [12, 13, 14, 15]]),
                3, 2, [(3, 1), (2, 0)],
                [[(3, 3), (3, 0), (1, 3)],
                 [(3, 2), (1, 2), (1, 0)],
                 [(2, 3), (2, 1), (0, 3)],
                 [(2, 2), (0, 2), (0, 0)],
                 [(1, 1)],
                 [(0, 1)]]),
                ])
def test_ParallelImageBlockMetricSeeder(image, n_samples,
                                        minimum_distance, excluded, expected):
    sb = seeder.ParallelImageBlockMetricSeeder(
            keep_fraction=1.0,
            seeder_grid_size=1,
            n_samples=n_samples,
            minimum_distance=minimum_distance)
    sb.select_seeds(image, sigma=None)
    for pixel in excluded:
        sb._excluded_pixels.add(pixel)
    provided = list(sb)
    assert len(provided) == len(expected)
    for p, e in zip(provided, expected):
        for a, b in zip(p, e):
            assert a == b
