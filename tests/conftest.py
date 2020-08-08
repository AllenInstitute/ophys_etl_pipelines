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
    """Fixture that allows parameterized mock motion correction files (*.csv)
    files to be generated.

    Old-style motion files do *not* contain a header row (yikes D:)
    You just have to know that they contain 9 columns with the following names:
    ["index", "x", "y", "a", "b", "c", "d", "e", "f"]

    New-style motion files may have more or fewer than 9 columns but will have
    a header row (which contains at least one alphabetical character [a-zA-Z]).
    """
    # Range of mock motion correction values to generate
    abs_value_bound = request.param.get("abs_value_bound", 5.0)
    default_col_names = ["index", "x", "y", "a", "b", "c", "d", "e", "f"]
    col_names = request.param.get("col_names", default_col_names)
    num_rows = request.param.get("num_rows", 15)
    required_x_values = request.param.get("required_x_values", [])
    required_y_values = request.param.get("required_y_values", [])
    deprecated_mode = request.param.get("deprecated_mode", False)
    random_seed = request.param.get("random_seed", 0)

    min_rows = max(len(required_x_values), len(required_y_values))
    num_rows = min_rows if min_rows > num_rows else num_rows

    seed(random_seed)
    motion_values = uniform(low=-abs_value_bound,
                            high=abs_value_bound,
                            size=(num_rows, 8))

    # Replace randomized motion values with required x and y values
    motion_values[:len(required_x_values), 0] = np.array(required_x_values)
    motion_values[:len(required_y_values), 1] = np.array(required_y_values)

    # Insert indices as first 'column' of array
    indices = list(range(num_rows))
    motion_data = np.insert(motion_values, 0, indices, axis=1)

    motion_corrected_df = pd.DataFrame(data=motion_data)
    motion_corrected_df.columns = col_names
    motion_correction_path = tmp_path / 'motion_correction.csv'

    if deprecated_mode:
        motion_corrected_df.to_csv(motion_correction_path,
                                   index=False, header=False)
    else:
        motion_corrected_df.to_csv(motion_correction_path,
                                   index=False)

    fixture_params = {'abs_value_bound': abs_value_bound,
                      'col_names': col_names,
                      'num_rows': num_rows,
                      'deprecated_mode': deprecated_mode,
                      'random_seed': random_seed,
                      'motion_data': motion_data}

    return motion_correction_path, fixture_params
