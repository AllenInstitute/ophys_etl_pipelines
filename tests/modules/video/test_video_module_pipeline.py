import pytest
import pathlib
import tempfile
import tifffile
import numpy as np
from itertools import product
from ophys_etl.modules.video.single_video import (
    VideoGenerator)

from ophys_etl.modules.video.side_by_side_video import (
    SideBySideVideoGenerator)


@pytest.mark.parametrize(
    'output_suffix, video_dtype, kernel_type',
    product(('.avi', '.mp4', '.tiff', '.tif'),
            ('uint8', 'uint16'),
            ('median', 'mean')))
def test_single_video_downsampling(
        tmpdir,
        video_path_fixture,
        output_suffix,
        video_dtype,
        kernel_type):
    """
    This is just a smoke test
    """
    tmp_dir = pathlib.Path(tmpdir)
    video_path = str(video_path_fixture.resolve().absolute())
    output_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir,
                                                suffix=output_suffix)[1])
    input_args = {'video_path': video_path,
                  'output_path': str(output_path.resolve().absolute()),
                  'kernel_size': 5,
                  'input_frame_rate_hz': 12.0,
                  'output_frame_rate_hz': 7.0,
                  'reticle': True,
                  'lower_quantile': 0.1,
                  'upper_quantile': 0.7,
                  'tmp_dir': str(tmp_dir.resolve().absolute()),
                  'quality': 6,
                  'speed_up_factor': 2,
                  'n_parallel_workers': 3,
                  'video_dtype': video_dtype,
                  'kernel_type': kernel_type}

    runner = VideoGenerator(input_data=input_args, args=[])
    runner.run()
    assert output_path.is_file()

    if output_path.suffix in ('.tiff', '.tif'):
        with tifffile.TiffFile(output_path, 'rb') as input_file:
            arr = input_file.pages[0].asarray()
            if video_dtype == 'uint8':
                assert arr.dtype == np.uint8
            else:
                assert arr.dtype == np.uint16


@pytest.mark.parametrize(
    'output_suffix, video_dtype, kernel_type',
    product(('.avi', '.mp4', '.tiff', '.tif'),
            ('uint8', 'uint16'),
            ('median', 'mean')))
def test_side_by_side_video_downsampling(
        tmpdir,
        video_path_fixture,
        output_suffix,
        video_dtype,
        kernel_type):
    """
    This is just a smoke test
    """
    tmp_dir = pathlib.Path(tmpdir)
    video_path = str(video_path_fixture.resolve().absolute())
    output_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir,
                                                suffix=output_suffix)[1])
    input_args = {'left_video_path': video_path,
                  'right_video_path': video_path,
                  'output_path': str(output_path.resolve().absolute()),
                  'kernel_size': 5,
                  'input_frame_rate_hz': 12.0,
                  'output_frame_rate_hz': 7.0,
                  'reticle': True,
                  'lower_quantile': 0.1,
                  'upper_quantile': 0.7,
                  'tmp_dir': str(tmp_dir.resolve().absolute()),
                  'quality': 6,
                  'speed_up_factor': 2,
                  'n_parallel_workers': 3,
                  'video_dtype': video_dtype,
                  'kernel_type': kernel_type}

    runner = SideBySideVideoGenerator(input_data=input_args, args=[])
    runner.run()
    assert output_path.is_file()

    if output_path.suffix in ('.tiff', '.tif'):
        with tifffile.TiffFile(output_path, 'rb') as input_file:
            arr = input_file.pages[0].asarray()
            if video_dtype == 'uint8':
                assert arr.dtype == np.uint8
            else:
                assert arr.dtype == np.uint16


def test_single_video_schema_error(
        tmpdir,
        video_path_fixture):
    """
    This is just a smoke test
    """
    output_suffix = '.jpg'
    video_dtype = 'uint8'
    kernel_type = 'mean'
    tmp_dir = pathlib.Path(tmpdir)
    video_path = str(video_path_fixture.resolve().absolute())
    output_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir,
                                                suffix=output_suffix)[1])
    input_args = {'video_path': video_path,
                  'output_path': str(output_path.resolve().absolute()),
                  'kernel_size': 5,
                  'input_frame_rate_hz': 12.0,
                  'output_frame_rate_hz': 7.0,
                  'reticle': True,
                  'lower_quantile': 0.1,
                  'upper_quantile': 0.7,
                  'tmp_dir': str(tmp_dir.resolve().absolute()),
                  'quality': 6,
                  'speed_up_factor': 2,
                  'n_parallel_workers': 3,
                  'video_dtype': video_dtype,
                  'kernel_type': kernel_type}

    with pytest.raises(ValueError, match='output_path must have'):
        _ = VideoGenerator(input_data=input_args, args=[])


def test_side_by_side_video_schema_error(
        tmpdir,
        video_path_fixture):
    output_suffix = '.jpg'
    video_dtype = 'uint8'
    kernel_type = 'mean'
    tmp_dir = pathlib.Path(tmpdir)
    video_path = str(video_path_fixture.resolve().absolute())
    output_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir,
                                                suffix=output_suffix)[1])
    input_args = {'left_video_path': video_path,
                  'right_video_path': video_path,
                  'output_path': str(output_path.resolve().absolute()),
                  'kernel_size': 5,
                  'input_frame_rate_hz': 12.0,
                  'output_frame_rate_hz': 7.0,
                  'reticle': True,
                  'lower_quantile': 0.1,
                  'upper_quantile': 0.7,
                  'tmp_dir': str(tmp_dir.resolve().absolute()),
                  'quality': 6,
                  'speed_up_factor': 2,
                  'n_parallel_workers': 3,
                  'video_dtype': video_dtype,
                  'kernel_type': kernel_type}

    with pytest.raises(ValueError, match='output_path must have'):
        _ = SideBySideVideoGenerator(input_data=input_args, args=[])
