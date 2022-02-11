from collections import defaultdict

import h5py
import numpy as np
import pytest

import imageio_ffmpeg as mpg
from ophys_etl.types import ExtractROI
import ophys_etl.utils.video_utils as transformations


@pytest.mark.parametrize(
        ("array, input_fps, output_fps, strategy, expected"),
        [
            (
                # average downsample video file with this dataset:
                np.array([
                    [[1, 1], [1, 1]],
                    [[2, 2], [2, 2]],
                    [[3, 3], [3, 3]],
                    [[4, 4], [4, 4]],
                    [[5, 5], [5, 5]],
                    [[6, 6], [6, 6]],
                    [[7, 7], [7, 7]]]),
                7, 2, 'average',
                np.array([
                    [[2.5, 2.5], [2.5, 2.5]],
                    [[6.0, 6.0], [6.0, 6.0]]])),
                ])
def test_video_downsample(
        array, input_fps, output_fps, strategy, expected, tmp_path):

    video_file = tmp_path / "sample_video_file.h5"
    with h5py.File(video_file, "w") as h5f:
        h5f.create_dataset('data', data=array)

    downsampled_video = transformations.downsample_h5_video(
            video_file,
            input_fps,
            output_fps,
            strategy)

    assert np.array_equal(downsampled_video, expected)


def compare_videos(encoded_video_path: str, expected_video: np.ndarray):
    """Compare an encoded video with its original source"""

    reader = mpg.read_frames(encoded_video_path,
                             pix_fmt="gray8",
                             bits_per_pixel=8)
    meta = reader.__next__()
    obt_nframes = int(np.round(meta['duration'] * meta['fps']))

    assert obt_nframes == len(expected_video)

    obt_frames = []
    for frame in reader:
        parsed_frame = np.frombuffer(frame, dtype='uint8')
        parsed_frame = parsed_frame.reshape(meta["size"][::-1])
        obt_frames.append(parsed_frame)
    obt_video = np.array(obt_frames)

    assert obt_video.shape == expected_video.shape
    # the default settings for imageio-ffmpeg are not lossless
    # so can't test for exact match
    np.testing.assert_allclose(obt_video, expected_video, atol=20)


@pytest.fixture
def raw_video_fixture(request):
    video_shape = request.param.get('video_shape', (16, 32))
    nframes = request.param.get('nframes', 25)
    fps = request.param.get('fps', 30)
    rng_seed = request.param.get('rng_seed', 0)

    rng = np.random.default_rng(rng_seed)

    raw_video = [rng.integers(0, 256, size=video_shape, dtype='uint8')
                 for _ in range(nframes)]

    result = {}
    result["raw_video"] = np.array(raw_video)
    result["fps"] = fps
    result["nframes"] = nframes

    return result


@pytest.mark.parametrize("raw_video_fixture", [
    # make the test video a size of at least 16x16
    # otherwise, need to mess with macro_block_size arg
    ({"video_shape": (16, 16)}),

    ({"video_shape": (32, 16)})
], indirect=["raw_video_fixture"])
def test_encode_video(raw_video_fixture, tmp_path):
    output_path = tmp_path / 'test_video.webm'

    fps = raw_video_fixture["fps"]
    expected_video = raw_video_fixture["raw_video"]

    transformations.encode_video(video=expected_video,
                                 output_path=output_path.as_posix(),
                                 fps=fps),

    compare_videos(output_path, expected_video)


@pytest.fixture
def encoded_videos_fixture(request, tmp_path):
    num_videos = request.param.get('num_videos', 2)
    video_shape = request.param.get('video_shape', (32, 32))
    nframes = request.param.get('nframes', 30)
    fps = request.param.get('fps', 30)

    rng = np.random.default_rng(0)
    test_videos = defaultdict(list)

    for i in range(num_videos):
        data = np.array([rng.integers(0, 256, size=video_shape, dtype='uint8')
                         for _ in range(nframes)])

        test_video_path = tmp_path / f"test_video_{i}.webm"
        transformations.encode_video(video=data,
                                     output_path=test_video_path.as_posix(),
                                     fps=fps)

        test_videos['raw_data'].append(data)
        test_videos['encoded_videos'].append(test_video_path)

    return test_videos


@pytest.mark.parametrize('padding, y0, x0, height, width',
                         [(5, 3, 2, 10, 12),
                          (10, 3, 2, 10, 12),
                          (5, 50, 50, 11, 23),
                          (10, 50, 50, 11, 23),
                          (5, 118, 50, 10, 12),
                          (10, 118, 50, 10, 12),
                          (5, 50, 118, 12, 10),
                          (10, 50, 118, 12, 10),
                          (5, 3, 50, 12, 13),
                          (10, 50, 3, 12, 13),
                          (5, 118, 118, 10, 10),
                          (10, 118, 118, 10, 10)])
def test_video_bounds_from_ROI(padding, x0, y0, height, width):

    roi = ExtractROI(x=x0, y=y0, height=height, width=width)
    x1 = x0 + width
    y1 = y0 + height

    origin, fov = transformations.video_bounds_from_ROI(roi,
                                                        (128, 128),
                                                        padding)
    assert fov[0] % 16 == 0
    assert fov[1] % 16 == 0
    assert fov[0] == fov[1]  # only considering cases that can give squares
    assert fov[0] >= max(height+2*padding, width+2*padding)
    assert origin[0] <= y0
    assert origin[1] <= x0
    assert origin[0]+fov[0] >= y1
    assert origin[1]+fov[1] >= x1
    assert origin[0] >= 0
    assert origin[1] >= 0
    assert origin[0]+fov[0] <= 128
    assert origin[1]+fov[1] <= 128
