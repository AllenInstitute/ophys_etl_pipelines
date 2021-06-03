import pytest
import numpy as np
import imageio
import copy
import h5py
import pathlib
import gc

from ophys_etl.modules.segmentation.qc_utils.video_utils import (
    thumbnail_video_from_array,
    ThumbnailVideo)    


@pytest.fixture
def example_video():
    rng = np.random.RandomState(16412)
    data = rng.randint(0, 255, (100, 30, 40)).astype(np.uint8)
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
