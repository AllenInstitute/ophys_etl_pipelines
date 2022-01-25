import pytest
import pathlib
import tempfile
from ophys_etl.modules.video.single_video import (
    VideoDownsampler)

from ophys_etl.modules.video.side_by_side_video import (
    SideBySideDownsampler)


@pytest.mark.parametrize('output_suffix', ('.avi', '.mp4'))
def test_single_video_downsampling(
        tmpdir,
        video_path_fixture,
        output_suffix):
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
                  'n_parallel_workers': 3}
    runner = VideoDownsampler(input_data=input_args, args=[])
    runner.run()
    assert output_path.is_file()


@pytest.mark.parametrize('output_suffix', ('.avi', '.mp4'))
def test_side_by_side_video_downsampling(
        tmpdir,
        video_path_fixture,
        output_suffix):
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
                  'n_parallel_workers': 3}
    runner = SideBySideDownsampler(input_data=input_args, args=[])
    runner.run()
    assert output_path.is_file()
