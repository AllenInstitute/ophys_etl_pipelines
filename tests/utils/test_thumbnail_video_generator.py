import pytest
import numpy as np
import h5py
import tempfile
import pathlib
import hashlib
from itertools import product

import ophys_etl.utils.thumbnail_video_utils as thumbnail_utils
from ophys_etl.types import ExtractROI
from ophys_etl.utils.thumbnail_video_generator import (
    VideoGenerator)


@pytest.fixture(scope='session')
def list_of_roi():
    """
    A list of ExtractROIs
    """
    output = []
    rng = np.random.default_rng(11231)
    for ii in range(10):
        x0 = int(rng.integers(0, 30))
        y0 = int(rng.integers(0, 30))
        width = int(rng.integers(4, 10))
        height = int(rng.integers(4, 10))
        mask = rng.integers(0, 2, size=(height, width)).astype(bool)

        # because np.ints are not JSON serializable
        real_mask = []
        for row in mask:
            this_row = []
            for el in row:
                if el:
                    this_row.append(True)
                else:
                    this_row.append(False)
            real_mask.append(this_row)

        if ii % 2 == 0:
            valid_roi = True
        else:
            valid_roi = False
        roi = ExtractROI(x=x0, width=width,
                         y=y0, height=height,
                         valid=valid_roi,
                         mask=real_mask,
                         id=ii)
        output.append(roi)
    return output


@pytest.fixture(scope='session')
def example_video(tmpdir_factory):
    """
    Write an example video to an HDF5 file; return the
    path to that file.
    """
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
    yield fname
    fname.unlink()
    tmpdir.rmdir()


@pytest.fixture
def example_roi():
    """an ExtractROI"""
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
    """test that VideoGenerator raises an exception when given
    a bogus file name"""
    with pytest.raises(RuntimeError, match='is not a file'):
        VideoGenerator('not_a_file.h5')


@pytest.mark.parametrize('origin, frame_shape, timesteps, as_array',
                         product((None, (5, 5)),
                                 (None, (16, 20)),
                                 (None, np.array([1, 4, 5, 17,
                                                  19, 23, 23, 25, 38])),
                                 (True, False)))
def test_get_thumbnail_video(tmpdir,
                             example_video,
                             origin,
                             frame_shape,
                             timesteps,
                             as_array):
    """
    Test that results of VideoGenerator and by-hand invocation of
    thumbnail_utils.thumbnail_video_from_path are identical
    """
    video_tmpdir = pathlib.Path(tmpdir) / 'video_temp'

    if as_array:
        with h5py.File(example_video, 'r') as in_file:
            generator = VideoGenerator(
                             video_data=in_file['data'][()],
                             tmp_dir=video_tmpdir)
    else:
        generator = VideoGenerator(video_path=example_video,
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

    if as_array:
        with h5py.File(example_video, 'r') as in_file:
            expected = thumbnail_utils.thumbnail_video_from_array(
                            in_file['data'][()],
                            origin,
                            frame_shape,
                            timesteps=timesteps,
                            tmp_dir=video_tmpdir,
                            fps=fps,
                            quality=quality)
    else:
        expected = thumbnail_utils.thumbnail_video_from_path(
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

    # to make sure we don't needlessly clutter CircleCIs
    # scratch space
    del thumbnail
    del expected
    del generator


@pytest.mark.parametrize('origin, frame_shape, timesteps, as_array',
                         product((None, (5, 5)),
                                 (None, (16, 20)),
                                 (None, np.array([1, 4, 5, 17,
                                                  19, 23, 23, 25, 38])),
                                 (True, False)))
def test_thumbnail_with_roi_list(
                             tmpdir,
                             example_video,
                             origin,
                             frame_shape,
                             timesteps,
                             list_of_roi,
                             as_array):
    """
    Just a smoketest that we can generate a by-hand thumbnail with ROIs
    displayed in it
    """
    video_tmpdir = pathlib.Path(tmpdir) / 'video_temp'

    if as_array:
        with h5py.File(example_video, 'r') as in_file:
            generator = VideoGenerator(
                            video_data=in_file['data'][()],
                            tmp_dir=video_tmpdir)
    else:
        generator = VideoGenerator(video_path=example_video,
                                   tmp_dir=video_tmpdir)

    fps = 11
    quality = 6

    thumbnail = generator.get_thumbnail_video(origin=origin,
                                              frame_shape=frame_shape,
                                              timesteps=timesteps,
                                              fps=fps,
                                              quality=quality,
                                              rois=list_of_roi)

    assert thumbnail.video_path.is_file()

    # to make sure we don't needlessly clutter CircleCIs
    # scratch space
    del thumbnail

    thumbnail = generator.get_thumbnail_video(origin=origin,
                                              frame_shape=frame_shape,
                                              timesteps=timesteps,
                                              fps=fps,
                                              quality=quality,
                                              rois=list_of_roi,
                                              valid_only=True)

    assert thumbnail.video_path.is_file()

    # to make sure we don't needlessly clutter CircleCIs
    # scratch space
    del thumbnail

    roi_lookup = {roi['id']: roi for roi in list_of_roi}

    thumbnail = generator.get_thumbnail_video(origin=origin,
                                              frame_shape=frame_shape,
                                              timesteps=timesteps,
                                              fps=fps,
                                              quality=quality,
                                              rois=roi_lookup)

    assert thumbnail.video_path.is_file()

    # to make sure we don't needlessly clutter CircleCIs
    # scratch space
    del thumbnail

    thumbnail = generator.get_thumbnail_video(origin=origin,
                                              frame_shape=frame_shape,
                                              timesteps=timesteps,
                                              fps=fps,
                                              quality=quality,
                                              rois=roi_lookup,
                                              valid_only=True)

    assert thumbnail.video_path.is_file()

    # to make sure we don't needlessly clutter CircleCIs
    # scratch space
    del thumbnail
    del generator


@pytest.mark.parametrize('roi_color, timesteps, padding, as_array',
                         product((None, (122, 201, 53)),
                                 (None, np.array([1, 4, 5, 17, 19,
                                                  23, 23, 25, 38])),
                                 (0, 5, 10),
                                 (True, False)))
def test_get_thumbnail_video_from_roi(
                             tmpdir,
                             example_video,
                             example_roi,
                             roi_color,
                             timesteps,
                             padding,
                             as_array):
    """
    Test that results of VideoGenerator and by-hand invocation of
    thumbnail_utils.thumbnail_video_from_ROI are identical
    """
    video_tmpdir = pathlib.Path(tmpdir) / 'video_temp'
    if as_array:
        with h5py.File(example_video, 'r') as in_file:
            video_data = in_file['data'][()]
        generator = VideoGenerator(
                             video_data=video_data,
                             tmp_dir=video_tmpdir)
    else:
        generator = VideoGenerator(video_path=example_video,
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

    if as_array:
        expected = thumbnail_utils.thumbnail_video_from_ROI(
                       video_data,
                       example_roi,
                       padding=padding,
                       roi_color=roi_color,
                       timesteps=timesteps,
                       tmp_dir=video_tmpdir,
                       fps=fps,
                       quality=quality)

    else:
        expected = thumbnail_utils.thumbnail_video_from_ROI(
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

    # to make sure we don't needlessly clutter CircleCIs
    # scratch space
    del thumbnail
    del expected
    del generator
