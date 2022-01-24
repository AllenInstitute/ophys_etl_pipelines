import h5py
import numpy as np
import tempfile
import pathlib
import pytest


@pytest.fixture(scope='session')
def ds_video_array_fixture():
    rng = np.random.default_rng(8712324)
    data = 2000.0*rng.random((53, 16, 16))
    return data


@pytest.fixture(scope='session')
def ds_video_path_fixture(ds_video_array_fixture, tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('ds_video_data'))
    file_path = tempfile.mkstemp(dir=tmpdir,
                                 prefix='ds_video_',
                                 suffix='.h5')[1]
    file_path = pathlib.Path(file_path)
    with h5py.File(file_path, 'w') as out_file:
        out_file.create_dataset('data', data=ds_video_array_fixture)

    yield file_path

    if file_path.is_file():
        file_path.unlink()
