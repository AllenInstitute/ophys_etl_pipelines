import pytest
import h5py
import numpy as np
import pathlib
import tempfile
import PIL.Image

from ophys_etl.modules.maximum_projection.utils import (
    generate_max_projection,
    scale_to_uint8)


from ophys_etl.modules.maximum_projection.__main__ import (
    MaximumProjectionRunner)


@pytest.fixture(scope='session')
def video_data_fixture():
    rng = np.random.default_rng(55123)
    data = rng.random((417, 22, 27))
    return data


@pytest.fixture(scope='session')
def video_path_fixture(video_data_fixture, tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp('maximum_projection_runner')
    video_path = tempfile.mkstemp(dir=tmpdir,
                                  suffix='.h5',
                                  prefix='video_')[1]
    with h5py.File(video_path, 'w') as out_file:
        out_file.create_dataset('data', data=video_data_fixture)
    yield video_path


@pytest.mark.parametrize('n_frames_at_once', [-1, 150])
def test_runner(tmpdir,
               video_data_fixture,
               video_path_fixture,
               n_frames_at_once):

    input_frame_rate = 6.0
    downsampled_frame_rate = 4.0
    median_kernel_size = 3
    n_processors = 3
    expected = generate_max_projection(
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

    runner = MaximumProjectionRunner(args=[], input_data=args)
    runner.run()

    assert pathlib.Path(image_path).is_file()
    assert pathlib.Path(full_path).is_file()
    with h5py.File(full_path, 'r') as in_file:
        actual = in_file['max_projection'][()]
    np.testing.assert_array_equal(actual, expected)

    expected = scale_to_uint8(expected)
    actual = np.array(PIL.Image.open(image_path, 'r'))
    np.testing.assert_array_equal(expected, actual)
