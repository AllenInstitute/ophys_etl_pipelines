from pathlib import Path
from typing import Tuple

import h5py
import pytest

import numpy as np
from scipy.sparse import coo_matrix


@pytest.fixture
def ophys_movie_fixture(tmp_path: Path, request) -> Tuple[Path, dict]:
    """Fixture that allows parametrized mock ophys vidoes (*.h5) to be
    generated.
    """
    params = request.param
    movie_name = params.get("movie_name", "ophys_movie.h5")
    movie_h5_key = params.get("movie_h5_key", "data")
    movie_shape = params.get("movie_shape", (10, 20, 20))
    movie_frames = params.get("movie_frames", np.random.rand(*movie_shape))
    experiment_dirname = params.get("experiment_dirname",
                                    "ophys_experiment_42")
    fixture_params = {"movie_name": movie_name, "movie_h5_key": movie_h5_key,
                      "movie_shape": movie_shape, "movie_frames": movie_frames,
                      "experiment_dirname": experiment_dirname}

    # Build a mock ophys movie dir structure similar to isilon production
    specimen_dir = tmp_path / "specimen_x"
    specimen_dir.mkdir()
    experiment_dir = specimen_dir / experiment_dirname
    experiment_dir.mkdir()
    processed_dir = experiment_dir / "processed"
    processed_dir.mkdir()

    ophys_movie_path = processed_dir / movie_name
    with h5py.File(ophys_movie_path, "w") as f:
        f.create_dataset(movie_h5_key, data=movie_frames)
    return (ophys_movie_path, fixture_params)


@pytest.fixture
def s2p_stat_fixture(tmp_path: Path, request) -> Tuple[Path, dict]:
    """Fixture that allows parametrized mock suite2p stat.npy files to
    be generated.
    """
    frame_shape = request.param.get("frame_shape", (20, 20))
    masks = request.param.get("masks", [np.random.rand(*frame_shape)
                                        for i in range(10)])
    fixture_params = {"frame_shape": frame_shape, "masks": masks}

    stats = []
    for dense_mask in masks:
        coo = coo_matrix(dense_mask, shape=frame_shape)
        stats.append({"lam": coo.data, "xpix": coo.col, "ypix": coo.row})

    stat_path = tmp_path / "stat.npy"
    np.save(stat_path, stats)
    return (stat_path, fixture_params)
