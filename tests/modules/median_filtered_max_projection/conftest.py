import pytest
import numpy as np
import h5py
import tempfile


@pytest.fixture(scope='session')
def video_data_fixture():
    rng = np.random.default_rng(55123)
    data = rng.random((417, 22, 27))
    return data


@pytest.fixture(scope='session')
def video_path_fixture(video_data_fixture, tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('maximum_projection_runner')
    video_path = tempfile.mkstemp(dir=tmpdir,
                                  suffix='.h5',
                                  prefix='video_')[1]
    with h5py.File(video_path, 'w') as out_file:
        out_file.create_dataset('data', data=video_data_fixture)
    yield video_path
