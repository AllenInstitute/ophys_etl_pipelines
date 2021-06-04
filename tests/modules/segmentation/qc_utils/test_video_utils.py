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
    scale_video_to_uint8,
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


def test_scale_video():

    data = np.array([[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0]])

    scaled = scale_video_to_uint8(data)
    assert scaled.dtype == np.uint8

    expected = np.array([[32, 64, 96, 128],
                         [159, 191, 223, 255]]).astype(np.uint8)

    np.testing.assert_array_equal(expected, scaled)

    scaled = scale_video_to_uint8(data, max_val=15)
    expected = np.array([[17, 34, 51, 68],
                         [85, 102, 119, 136]]).astype(np.uint8)

    np.testing.assert_array_equal(expected, scaled)


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

    y0 = 20
    height = 7
    x0 = 10
    width = 8

    bdry_pix = []
    for row in (3,4):
        for col in range(1,6):
            bdry_pix.append((row+y0, col+x0))
    for row in range(3,5):
        for col in (1,5):
            bdry_pix.append((row+y0, col+x0))


    roi = ExtractROI(y=y0,
                     height=height,
                     x=x0,
                     width=width,
                     valid=True,
                     mask=[list(row) for row in mask])

    thumbnail = thumbnail_video_from_ROI(
                    example_video,
                    roi,
                    tmp_dir=pathlib.Path(tmpdir),
                    quality=10)

    rowmin = thumbnail.origin[0]
    rowmax = thumbnail.origin[0]+thumbnail.frame_shape[0]
    colmin = thumbnail.origin[1]
    colmax = thumbnail.origin[1]+thumbnail.frame_shape[1]
    assert rowmin <= y0
    assert rowmax >= y0+height
    assert colmin <= x0
    assert colmax >= x0+width

    assert thumbnail.video_path.is_file()

    read_data = imageio.mimread(thumbnail.video_path)
    assert read_data[0].shape == (thumbnail.frame_shape[0],
                                  thumbnail.frame_shape[1],
                                  3)

    # now with color
    example_video[:,:,:] = 0
    thumbnail = thumbnail_video_from_ROI(
                    example_video,
                    roi,
                    roi_color=(0, 255, 0),
                    tmp_dir=pathlib.Path(tmpdir),
                    quality=7)

    rowmin = thumbnail.origin[0]
    rowmax = thumbnail.origin[0]+thumbnail.frame_shape[0]
    colmin = thumbnail.origin[1]
    colmax = thumbnail.origin[1]+thumbnail.frame_shape[1]
    assert rowmin <= 20
    assert rowmax >= 27
    assert colmin <= 10
    assert colmax >= 18

    assert thumbnail.video_path.is_file()
