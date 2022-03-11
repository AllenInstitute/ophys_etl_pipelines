import pytest
import pathlib
import numpy as np
import tempfile
import h5py
from ophys_etl.types import ExtractROI


@pytest.fixture(scope='session')
def video_file_fixture(tmp_path_factory):
    """
    Create an HDF5 file with 'video_data'.
    Yield the path to that file.
    """
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('artifact_video'))
    rng = np.random.default_rng(7213)
    video_path = tempfile.mkstemp(dir=tmp_dir, suffix='.h5')[1]
    video_path = pathlib.Path(video_path)
    with h5py.File(video_path, 'w') as out_file:
        out_file.create_dataset(
                'video_data',
                data=rng.integers(0, 255, (100, 64, 64)).astype(np.uint8),
                chunks=(10, 16, 16))
    yield video_path
    if video_path.is_file():
        video_path.unlink()


@pytest.fixture(scope='session')
def extract_roi_list_fixture():
    """
    Return a list of ExtractROIs
    """
    rng = np.random.default_rng(5321)
    roi_list = []

    mask = rng.integers(0, 2, (10, 10)).astype(bool)
    roi_list.append(ExtractROI(id=0,
                               x=4, y=3,
                               width=mask.shape[1],
                               height=mask.shape[0],
                               valid=True,
                               mask=[list(v) for v in mask]))

    mask = rng.integers(0, 2, (7, 12)).astype(bool)
    roi_list.append(ExtractROI(id=1,
                               x=10, y=3,
                               width=mask.shape[1],
                               height=mask.shape[0],
                               valid=True,
                               mask=[list(v) for v in mask]))

    mask = rng.integers(0, 2, (7, 12)).astype(bool)
    roi_list.append(ExtractROI(id=2,
                               x=1, y=1,
                               width=mask.shape[1],
                               height=mask.shape[0],
                               valid=True,
                               mask=[list(v) for v in mask]))

    mask = rng.integers(0, 2, (7, 12)).astype(bool)
    roi_list.append(ExtractROI(id=3,
                               x=12, y=4,
                               width=mask.shape[1],
                               height=mask.shape[0],
                               valid=True,
                               mask=[list(v) for v in mask]))

    return roi_list


@pytest.fixture(scope='session')
def roi_with_string_as_id():
    rng = np.random.default_rng(5321)
    mask = rng.integers(0, 2, (7, 12)).astype(bool)
    return ExtractROI(id='aaa',
                      x=9, y=6,
                      width=mask.shape[1],
                      height=mask.shape[0],
                      valid=True,
                      mask=[list(v) for v in mask])
