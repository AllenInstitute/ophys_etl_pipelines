import pytest
import numpy as np
import imageio
import copy
import h5py
import pathlib
import tempfile
import gc

from ophys_etl.types import ExtractROI

from ophys_etl.modules.segmentation.qc_utils.video_utils import (
    thumbnail_video_from_array,
    thumbnail_video_from_path,
    _thumbnail_video_from_ROI_array,
    _thumbnail_video_from_ROI_path,
    thumbnail_video_from_ROI,
    scale_video_to_uint8,
    video_bounds_from_ROI,
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


@pytest.fixture
def example_unnormalized_rgb_video():
    rng = np.random.RandomState(6125321)
    data = rng.randint(0, 700, (100, 50, 40, 3))
    return data


def test_thumbnail_video(tmpdir, example_video):
    """
    Just test that ThumbnailVideo can write the video
    and set its properties correctly
    """

    video_path = tempfile.mkstemp(dir=tmpdir, suffix='.mp4')[1]
    thumbnail = ThumbnailVideo(example_video,
                               pathlib.Path(video_path),
                               (111, 222),
                               quality=6,
                               fps=22)

    assert thumbnail.video_path.is_file()
    read_data = imageio.mimread(thumbnail.video_path)
    assert len(read_data) == example_video.shape[0]
    assert thumbnail.frame_shape == (example_video.shape[1],
                                     example_video.shape[2])

    assert thumbnail.origin == (111, 222)
    assert thumbnail.timesteps is None

    # check that the video is deleted when the thumbnail is deleted
    test_path = copy.deepcopy(thumbnail.video_path)
    del thumbnail
    gc.collect()
    assert not test_path.exists()

    # test non-None timesteps
    video_path = tempfile.mkstemp(dir=tmpdir, suffix='.mp4')[1]
    thumbnail = ThumbnailVideo(example_video,
                               pathlib.Path(video_path),
                               (111, 222),
                               quality=6,
                               fps=22,
                               timesteps=np.arange(450, 550))

    assert thumbnail.video_path.is_file()
    read_data = imageio.mimread(thumbnail.video_path)
    assert len(read_data) == example_video.shape[0]
    assert thumbnail.frame_shape == (example_video.shape[1],
                                     example_video.shape[2])

    assert thumbnail.origin == (111, 222)
    np.testing.assert_array_equal(thumbnail.timesteps,
                                  np.arange(450, 550))

    # check that the video is deleted when the thumbnail is deleted
    test_path = copy.deepcopy(thumbnail.video_path)
    del thumbnail
    gc.collect()
    assert not test_path.exists()


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


def test_video_bounds_from_ROI():
    roi = ExtractROI(x=15,
                     width=22,
                     y=19,
                     height=13)

    origin, fov = video_bounds_from_ROI(roi, (128, 128))
    assert fov == (32, 32)
    assert origin[0] <= 19
    assert origin[1] <= 15
    assert origin[0]+fov[0] >= 32
    assert origin[1]+fov[1] >= 37
    assert origin[0] >= 0
    assert origin[1] >= 0
    assert origin[0]+fov[0]<=128
    assert origin[1]+fov[1]<=128

    # constrained dimensions
    roi = ExtractROI(x=2,
                     width=4,
                     y=3,
                     height=6)

    origin, fov = video_bounds_from_ROI(roi, (10, 10))
    assert origin[0] <= 3
    assert origin[1] <= 2
    assert origin[0]+fov[0] >= 9
    assert origin[1]+fov[1] >= 6
    assert origin[0] >= 0
    assert origin[1] >= 0
    assert origin[0]+fov[0]<=10
    assert origin[1]+fov[1]<=10

    # constrained dimensions
    roi = ExtractROI(x=120,
                     width=4,
                     y=121,
                     height=6)

    origin, fov = video_bounds_from_ROI(roi, (128, 128))
    assert fov == (16, 16)
    assert origin[0] <= 121
    assert origin[1] <= 120
    assert origin[0]+fov[0] >= 127
    assert origin[1]+fov[1] >= 124
    assert origin[0] >= 0
    assert origin[1] >= 0
    assert origin[0]+fov[0]<=128
    assert origin[1]+fov[1]<=128


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

    thumbnail = _thumbnail_video_from_ROI_array(
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
    thumbnail = _thumbnail_video_from_ROI_array(
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


@pytest.mark.parametrize("custom_max_val", [None, 900])
def test_thumbnail_from_path(tmpdir,
                             example_unnormalized_rgb_video,
                             custom_max_val):
    """
    Test thumbnail_from_path by comparing output to result
    from thumbnail_from_array
    """

    # write video to a tempfile
    h5_fname = tempfile.mkstemp(dir=tmpdir, prefix='input_video_',
                                suffix='.h5')[1]
    with h5py.File(h5_fname, 'w') as out_file:
        out_file.create_dataset('data', data=example_unnormalized_rgb_video)

    sub_video = example_unnormalized_rgb_video[:, 18:30, 14:29, :]
    sub_video = scale_video_to_uint8(sub_video, max_val=custom_max_val)

    control_video = thumbnail_video_from_array(
                       sub_video,
                       (0,0),
                       (12, 15),
                       tmp_dir=pathlib.Path(tmpdir))

    test_video = thumbnail_video_from_path(
                     pathlib.Path(h5_fname),
                     (18, 14),
                     (12, 15),
                     tmp_dir=pathlib.Path(tmpdir),
                     max_val=custom_max_val)

    assert test_video.origin == (18, 14)
    assert test_video.frame_shape == (12, 15)
    control_data = imageio.mimread(control_video.video_path)
    test_data = imageio.mimread(test_video.video_path)
    assert len(control_data) == len(test_data)
    for ii in range(len(control_data)):
        np.testing.assert_array_equal(control_data[ii], test_data[ii])


@pytest.mark.parametrize("custom_max_val,roi_color",
                         [(None, None),
                          (None, (255, 0, 0)),
                          (900, None),
                          (900, (255, 0, 0))])
def test_thumbnail_from_roi_and_path(tmpdir,
                                     example_unnormalized_rgb_video,
                                     custom_max_val,
                                     roi_color):
    """
    Test _thumbnail_from_ROI_path by comparing output to result
    from _thumbnail_from_ROI_array
    """

    mask = np.zeros((12, 15), dtype=bool)
    mask[2:10, 3:13] = True

    roi = ExtractROI(x=14, width=15,
                     y=18, height=12,
                     mask=[list(i) for i in mask])

    # write video to a tempfile
    h5_fname = tempfile.mkstemp(dir=tmpdir, prefix='input_video_',
                                suffix='.h5')[1]
    with h5py.File(h5_fname, 'w') as out_file:
        out_file.create_dataset('data', data=example_unnormalized_rgb_video)

    if custom_max_val is None:
        mx = example_unnormalized_rgb_video[:, 18:30, 14:29, :].max()
    else:
        mx = custom_max_val

    normalized_video = scale_video_to_uint8(example_unnormalized_rgb_video,
                                            max_val=mx)

    control_video = _thumbnail_video_from_ROI_array(
                       normalized_video,
                       roi,
                       roi_color=roi_color,
                       tmp_dir=pathlib.Path(tmpdir))

    test_video = _thumbnail_video_from_ROI_path(
                     pathlib.Path(h5_fname),
                     roi,
                     roi_color=roi_color,
                     tmp_dir=pathlib.Path(tmpdir),
                     max_val=custom_max_val)

    control_data = imageio.mimread(control_video.video_path)
    test_data = imageio.mimread(test_video.video_path)
    assert test_video.origin == control_video.origin
    assert test_video.frame_shape == control_video.frame_shape
    assert len(control_data) == len(test_data)
    for ii in range(len(control_data)):
        np.testing.assert_array_equal(control_data[ii], test_data[ii])


def test_generic_generation_from_ROI(tmpdir, example_video):
    """
    Just smoketest thumbnail_video_from_ROI
    """
    mask = np.zeros((12, 15), dtype=bool)
    mask[2:10, 3:13] = True

    roi = ExtractROI(x=14, width=15,
                     y=18, height=12,
                     mask=[list(i) for i in mask])

    th = thumbnail_video_from_ROI(example_video.astype(np.uint8),
                                  roi,
                                  roi_color=(0, 255, 0),
                                  timesteps=None,
                                  file_path=None,
                                  tmp_dir=pathlib.Path(tmpdir),
                                  quality=7)

    base_fname = tempfile.mkstemp(dir=tmpdir, suffix='.h5')[1]
    with h5py.File(base_fname, 'w') as out_file:
        out_file.create_dataset('data',
                                data=example_video)

    th = thumbnail_video_from_ROI(pathlib.Path(base_fname),
                                  roi,
                                  roi_color=(0, 255, 0),
                                  timesteps=None,
                                  file_path=None,
                                  tmp_dir=pathlib.Path(tmpdir),
                                  quality=7)
