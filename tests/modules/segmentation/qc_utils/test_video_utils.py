import pytest
import numpy as np
import imageio
import copy
import h5py
import pathlib
import gc

from ophys_etl.types import ExtractROI

from ophys_etl.modules.segmentation.qc_utils.video_utils import (
    thumbnail_video_from_array,
    thumbnail_video_from_ROI,
    ThumbnailVideo)    


@pytest.fixture
def example_video():
    rng = np.random.RandomState(16412)
    data = rng.randint(0, 100, (100, 30, 40)).astype(np.uint8)
    for ii in range(100):
        data[ii,::,:] = ii
    return data


@pytest.fixture
def example_rgb_video():
    rng = np.random.RandomState(16412)
    data = rng.randint(0, 100, (100, 30, 40, 3)).astype(np.uint8)
    for ii in range(100):
        data[ii,::,:] = ii
    return data


def test_thumbnail_from_array(tmpdir, example_video):

    th_video = thumbnail_video_from_array(example_video,
                                          (11, 3),
                                          (16, 32))

    assert type(th_video) == ThumbnailVideo
    assert th_video.video_path.is_file()
    assert th_video.origin == (11, 3)
    assert th_video.frame_shape == (16, 32)
    assert th_video.timesteps is None

    read_data = imageio.mimread(th_video.video_path)

    assert len(read_data) == example_video.shape[0]
    assert read_data[0].shape == (16, 32, 3)

    # cannot to bitwise comparison of input to read data;
    # mp4 compression leads to slight differences

    file_path = str(th_video.video_path)
    file_path = pathlib.Path(file_path)

    del th_video
    gc.collect()

    assert not file_path.exists()


def test_thumbnail_from_rgb_array(tmpdir, example_rgb_video):

    th_video = thumbnail_video_from_array(example_rgb_video,
                                          (11, 3),
                                          (16, 32))

    assert type(th_video) == ThumbnailVideo
    assert th_video.video_path.is_file()
    assert th_video.origin == (11, 3)
    assert th_video.frame_shape == (16, 32)
    assert th_video.timesteps is None

    read_data = imageio.mimread(th_video.video_path)

    assert len(read_data) == example_rgb_video.shape[0]
    assert read_data[0].shape == (16, 32, 3)

    # cannot to bitwise comparison of input to read data;
    # mp4 compression leads to slight differences

    file_path = str(th_video.video_path)
    file_path = pathlib.Path(file_path)

    del th_video
    gc.collect()

    assert not file_path.exists()


def test_thumbnail_from_roi(tmpdir, example_video):


    mask = np.zeros((7,8), dtype=bool)
    mask[3:5, 1:6] = True

    roi = ExtractROI(y=20,
                     height=7,
                     x=10,
                     width=8,
                     valid=True,
                     mask=[list(row) for row in mask])

    thumbnail = thumbnail_video_from_ROI(
                    example_video,
                    roi,
                    tmp_dir=pathlib.Path(tmpdir))

    rowmin = thumbnail.origin[0]
    rowmax = thumbnail.origin[0]+thumbnail.frame_shape[0]
    colmin = thumbnail.origin[1]
    colmax = thumbnail.origin[1]+thumbnail.frame_shape[1]
    assert rowmin <= 20
    assert rowmax >= 27
    assert colmin <= 10
    assert colmax >= 18

    assert thumbnail.video_path.is_file()

    # now with color
    thumbnail = thumbnail_video_from_ROI(
                    example_video,
                    roi,
                    roi_color=(0, 255, 0),
                    tmp_dir=pathlib.Path(tmpdir))

    rowmin = thumbnail.origin[0]
    rowmax = thumbnail.origin[0]+thumbnail.frame_shape[0]
    colmin = thumbnail.origin[1]
    colmax = thumbnail.origin[1]+thumbnail.frame_shape[1]
    assert rowmin <= 20
    assert rowmax >= 27
    assert colmin <= 10
    assert colmax >= 18

    assert thumbnail.video_path.is_file()
