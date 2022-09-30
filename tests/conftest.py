from typing import Union
from pathlib import Path
from typing import Tuple

import h5py
import pandas as pd
from numpy.random import seed, uniform
import pytest
import json
import shutil

import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.utils.roi_masks import RoiMask


class HelperFunctions(object):
    @staticmethod
    def clean_up_dir(tmpdir: Union[str, Path]):
        """
        Attempt to clean up all of the files in a specified
        directory. If a file cannot be deleted, just catch the
        exception and move on.

        Attempt to remove the dir after cleanup.
        """
        tmpdir = Path(tmpdir)
        path_list = [n for n in tmpdir.rglob('*')]
        for this_path in path_list:
            if this_path.is_file():
                try:
                    this_path.unlink()
                except Exception:
                    pass
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass


@pytest.fixture
def helper_functions():
    """
    See solution to making helper functions available across
    a pytest module in
    https://stackoverflow.com/questions/33508060/create-and-import-helper-functions-in-tests-without-creating-packages-in-test-di
    """
    return HelperFunctions


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


@pytest.fixture()
def trace_file_fixture(tmp_path: Path, request) -> Tuple[Path, dict]:
    """Fixture that allows parametrized optical physiology trace files
    (*.h5) to be generated."""

    trace_name = request.param.get("trace_filename", "mock_trace.h5")

    trace_data = request.param.get("trace_data",
                                   np.arange(100).reshape((5, 20)))
    trace_data_key = request.param.get("trace_data_key", "data")

    trace_names = request.param.get("trace_names", ['0', '4', '1', '3', '2'])
    trace_names_key = request.param.get("trace_names_key", "roi_names")

    fixture_params = {"trace_data": trace_data,
                      "trace_data_key": trace_data_key,
                      "trace_names": trace_names,
                      "trace_names_key": trace_names_key}

    trace_path = tmp_path / trace_name
    with h5py.File(trace_path, "w") as f:
        f[trace_data_key] = trace_data
        formatted_trace_names = np.array(trace_names).astype(np.string_)
        f.create_dataset(trace_names_key, data=formatted_trace_names)

    return trace_path, fixture_params


@pytest.fixture
def image_dims():
    return {
        'width': 100,
        'height': 100
    }


@pytest.fixture
def motion_border():
    return [5.0, 5.0, 5.0, 5.0]


@pytest.fixture
def motion_border_dict():
    return {"x0": 5.0, "x1": 5.0, "y0": 5.0, "y1": 5.0}


@pytest.fixture
def roi_mask_list(image_dims, motion_border):
    base_pixels = np.argwhere(np.ones((10, 10)))

    masks = []
    for ii in range(10):
        pixels = base_pixels + ii * 10
        masks.append(RoiMask.create_roi_mask(
            image_dims['width'],
            image_dims['height'],
            motion_border,
            pix_list=pixels,
            label=str(ii),
            mask_group=-1
        ))

    return masks


@pytest.fixture
def ophys_plane_data_fixture():
    this_dir = Path(__file__)
    resource_dir = this_dir.parent / 'resources'
    json_path = resource_dir / 'ophys_plane_instantiation_data.json'
    with open(json_path, 'rb') as in_file:
        data = json.load(in_file)
    return data
