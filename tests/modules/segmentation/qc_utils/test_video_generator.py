import pytest
import numpy as np
import h5py
import tempfile
import pathlib
import hashlib
from itertools import product

import ophys_etl.modules.segmentation.qc_utils.video_utils as video_utils
from ophys_etl.types import ExtractROI
from ophys_etl.modules.segmentation.qc_utils.video_generator import (
    VideoGenerator)
from ophys_etl.modules.segmentation.qc_utils.video_display_generator import (
    VideoDisplayGenerator)


@pytest.fixture(scope='session')
def example_video(tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('eg_video'))
    rng = np.random.RandomState(121311)
    nt = 100
    nrows = 50
    ncols = 50
    data = rng.randint(0, 1000, (nt, nrows, ncols))
    fname = tempfile.mkstemp(dir=tmpdir,
                             prefix='video_generator_example_video_',
                             suffix='.h5')[1]
    with h5py.File(fname, 'w') as out_file:
        out_file.create_dataset('data', data=data)
    fname = pathlib.Path(fname)
    try:
        yield fname
    finally:
        fname.unlink()
        tmpdir.rmdir()


@pytest.fixture
def example_roi():
    x = 11
    y = 7
    height = 20
    width = 35
    rng = np.random.RandomState(8123)
    mask = rng.randint(0, 2, (height, width)).astype(bool)
    roi = ExtractROI(x=x,
                     y=y,
                     height=height,
                     width=width,
                     mask=mask,
                     valid=True,
                     id=999)
    return roi


def compare_hashes(fname0, fname1):
    """
    Assert that two files have the same md5 checksum
    """
    hash0 = hashlib.md5()
    hash1 = hashlib.md5()
    with open(fname0, 'rb') as in_file:
        data = in_file.read()
        hash0.update(data)
    with open(fname1, 'rb') as in_file:
        data = in_file.read()
        hash1.update(data)
    assert hash0.hexdigest() == hash1.hexdigest()


def test_video_generator_exception():
    with pytest.raises(RuntimeError, match='is not a file'):
        VideoGenerator('not_a_file.h5')


@pytest.mark.parametrize('origin, frame_shape, timesteps',
                         product((None, (5, 5)),
                                 (None, (16, 20)),
                                 (None, np.array([1, 4, 5, 17,
                                                  19, 23, 23, 25, 38]))))
def test_get_thumbnail_video(tmpdir,
                             example_video,
                             origin,
                             frame_shape,
                             timesteps):
    """
    Test that results of VideoGenerator and by-hand invocation of
    video_utils.thumbnail_video_from_path are identical
    """
    video_tmpdir = pathlib.Path(tmpdir) / 'video_temp'
    generator = VideoGenerator(example_video,
                               tmp_dir=video_tmpdir)

    fps = 11
    quality = 6

    thumbnail = generator.get_thumbnail_video(origin=origin,
                                              frame_shape=frame_shape,
                                              timesteps=timesteps,
                                              fps=fps,
                                              quality=quality)

    assert thumbnail.video_path.is_file()

    if origin is None:
        origin = (0, 0)
    if frame_shape is None:
        with h5py.File(example_video, 'r') as in_file:
            frame_shape = in_file['data'].shape

    expected = video_utils.thumbnail_video_from_path(
                    pathlib.Path(example_video),
                    origin,
                    frame_shape,
                    timesteps=timesteps,
                    tmp_dir=video_tmpdir,
                    fps=fps,
                    quality=quality,
                    min_max=generator.min_max)

    assert expected.video_path.is_file()
    assert not expected.video_path == thumbnail.video_path
    compare_hashes(expected.video_path, thumbnail.video_path)

    del thumbnail
    del expected
    del generator


@pytest.mark.parametrize('roi_color, timesteps, padding',
                         product((None, (122, 201, 53)),
                                 (None, np.array([1, 4, 5, 17, 19,
                                                  23, 23, 25, 38])),
                                 (0, 5, 10)))
def test_get_thumbnail_video_from_roi(
                             tmpdir,
                             example_video,
                             example_roi,
                             roi_color,
                             timesteps,
                             padding):
    """
    Test that results of VideoGenerator and by-hand invocation of
    video_utils.thumbnail_video_from_ROI are identical
    """
    video_tmpdir = pathlib.Path(tmpdir) / 'video_temp'
    generator = VideoGenerator(example_video,
                               tmp_dir=video_tmpdir)

    fps = 11
    quality = 6

    thumbnail = generator.get_thumbnail_video_from_roi(
                                  roi=example_roi,
                                  padding=padding,
                                  roi_color=roi_color,
                                  timesteps=timesteps,
                                  fps=fps,
                                  quality=quality)

    assert thumbnail.video_path.is_file()

    expected = video_utils.thumbnail_video_from_ROI(
                       pathlib.Path(example_video),
                       example_roi,
                       padding=padding,
                       roi_color=roi_color,
                       timesteps=timesteps,
                       tmp_dir=video_tmpdir,
                       fps=fps,
                       quality=quality,
                       min_max=generator.min_max)

    assert expected.video_path.is_file()
    assert not expected.video_path == thumbnail.video_path
    compare_hashes(expected.video_path, thumbnail.video_path)

    del thumbnail
    del expected
    del generator


def test_video_display_generator(tmpdir, example_video, example_roi):
    """
    Test that VideoDisplayGenerator creates symlinks to thumbnail
    videos in the expected place
    """

    video_tmpdir = pathlib.Path(tmpdir) / 'video_temp'
    generator = VideoGenerator(example_video,
                               tmp_dir=video_tmpdir)

    fps = 11
    quality = 6

    thumbnail = generator.get_thumbnail_video_from_roi(
                                  roi=example_roi,
                                  roi_color=(255, 0, 0),
                                  timesteps=None,
                                  fps=fps,
                                  quality=quality)

    # create a test class so that we do not put symlinks
    # in the path from which the tests are being run
    class TestDisplayGenerator(VideoDisplayGenerator):
        def __init__(self):
            self.this_dir = pathlib.Path(tmpdir)
            self.tmp_dir = self.this_dir/'silly/path/to/files'
            self.files_written = []

    display_generator = TestDisplayGenerator()
    params = display_generator.display_video(thumbnail)
    sym_path = pathlib.Path(tmpdir) / pathlib.Path(params['data'])
    assert sym_path.is_file()
    assert sym_path.resolve() == thumbnail.video_path.resolve()
    assert sym_path.is_symlink()
    assert sym_path.absolute() != thumbnail.video_path.absolute()

    del thumbnail
    del generator
