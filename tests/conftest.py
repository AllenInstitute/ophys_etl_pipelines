from pathlib import Path
from typing import Tuple

import h5py
import pandas as pd
from numpy.random import seed, uniform
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
    return ophys_movie_path, fixture_params


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
    return stat_path, fixture_params


@pytest.fixture()
def motion_correction_fixture(tmp_path: Path, request) -> Tuple[Path, dict]:
    """Fixture that allows parameterized mock motion correction video files
    to be generated
    """
    abs_value_bound = request.param.get("abs_value_bound", 5.0)
    motion_correction_rows = request.param.get("motion_correction_rows", 15)
    included_values_x = request.param.get("included_values_x", [])
    included_values_y = request.param.get("included_values_y", [])
    random_seed = request.param.get("random_seed", 0)

    fixture_params = {'abs_value_bound': abs_value_bound,
                      'motion_correction_rows': motion_correction_rows,
                      'random_seed': random_seed}

    seed(random_seed)
    x_correction_values = uniform(-abs_value_bound, abs_value_bound,
                                  motion_correction_rows)
    y_correction_values = uniform(-abs_value_bound, abs_value_bound,
                                  motion_correction_rows)
    x_correction_values = np.append(x_correction_values, included_values_x)
    y_correction_values = np.append(y_correction_values, included_values_y)
    '''
    for x_value in included_values_x:
        x_correction_values = np.append(x_correction_values, x_value)
    for y_value  in included_values_y:
        y_correction_values = np.append(y_correction_values, y_value)'''

    motion_correction_data = {
        'x': x_correction_values,
        'y': y_correction_values
    }

    motion_correction_path = tmp_path / 'motion_correction.csv'
    motion_corrected_df = pd.DataFrame.from_dict(motion_correction_data)

    motion_corrected_df.to_csv(motion_correction_path)
    return motion_correction_path, fixture_params
