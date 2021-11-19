import pytest
import h5py
import numpy as np
import pathlib
import tempfile
import PIL.Image

from ophys_etl.utils.array_utils import normalize_array

from ophys_etl.modules.median_filtered_max_projection.utils import (
    median_filtered_max_projection_from_array)


from ophys_etl.modules.median_filtered_max_projection.__main__ import (
    MedianFilteredMaxProjectionRunner)


@pytest.mark.parametrize('n_frames_at_once', [-1, 150])
def test_runner(tmpdir,
                video_data_fixture,
                video_path_fixture,
                n_frames_at_once):

    input_frame_rate = 6.0
    downsampled_frame_rate = 4.0
    median_kernel_size = 3
    n_processors = 3
    expected = median_filtered_max_projection_from_array(
                    video_data_fixture,
                    input_frame_rate,
                    downsampled_frame_rate,
                    median_kernel_size,
                    n_processors)

    args = dict()
    args['video_path'] = video_path_fixture
    args['input_frame_rate'] = input_frame_rate
    args['downsampled_frame_rate'] = downsampled_frame_rate
    args['n_parallel_workers'] = n_processors
    args['median_filter_kernel_size'] = median_kernel_size
    args['n_frames_at_once'] = n_frames_at_once

    image_path = tempfile.mkstemp(dir=tmpdir,
                                  prefix='image_',
                                  suffix='.png')[1]

    full_path = tempfile.mkstemp(dir=tmpdir,
                                 prefix='data_',
                                 suffix='.h5')[1]

    args['image_path'] = image_path
    args['full_output_path'] = full_path

    runner = MedianFilteredMaxProjectionRunner(args=[], input_data=args)
    runner.run()

    assert pathlib.Path(image_path).is_file()
    assert pathlib.Path(full_path).is_file()
    with h5py.File(full_path, 'r') as in_file:
        actual = in_file['max_projection'][()]
    np.testing.assert_array_equal(actual, expected)

    expected = normalize_array(expected)
    actual = np.array(PIL.Image.open(image_path, 'r'))
    np.testing.assert_array_equal(expected, actual)


def test_maximum_runner_exceptions(video_path_fixture, tmpdir):
    input_frame_rate = 6.0
    downsampled_frame_rate = 4.0
    median_kernel_size = 3
    n_processors = 3

    args = dict()
    args['video_path'] = video_path_fixture
    args['input_frame_rate'] = input_frame_rate
    args['downsampled_frame_rate'] = downsampled_frame_rate
    args['n_parallel_workers'] = n_processors
    args['median_filter_kernel_size'] = median_kernel_size
    args['n_frames_at_once'] = -1

    image_path = tempfile.mkstemp(dir=tmpdir,
                                  prefix='image_',
                                  suffix='.jpg')[1]

    full_path = tempfile.mkstemp(dir=tmpdir,
                                 prefix='data_',
                                 suffix='.h5')[1]

    args['image_path'] = image_path
    args['full_output_path'] = full_path

    with pytest.raises(ValueError, match="path to a .png file"):
        MedianFilteredMaxProjectionRunner(args=[], input_data=args)

    image_path = tempfile.mkstemp(dir=tmpdir,
                                  prefix='image_',
                                  suffix='.png')[1]

    full_path = tempfile.mkstemp(dir=tmpdir,
                                 prefix='data_',
                                 suffix='.txt')[1]

    args['image_path'] = image_path
    args['full_output_path'] = full_path

    with pytest.raises(ValueError, match="path to a .h5 file"):
        MedianFilteredMaxProjectionRunner(args=[], input_data=args)
