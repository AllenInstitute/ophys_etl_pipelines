import pytest
import numpy as np
from unittest.mock import MagicMock

from ophys_etl.modules.segmentation.denoise import averaging


@pytest.fixture
def random_video(request):
    # random values, used to smoke test functions
    data = np.random.randint(0, 2**15,
                             size=request.param["video_shape"],
                             dtype='uint16')
    return data


@pytest.mark.parametrize(
        "random_video",
        [
            {"video_shape": (100, 20, 20)}],
        indirect=["random_video"])
@pytest.mark.parametrize("filter_type", ["uniform", "gaussian"])
@pytest.mark.parametrize("size", [1, 10])
def test_averaging(random_video, filter_type, size):
    averaged = averaging.temporal_filter1d(video=random_video,
                                           size=size,
                                           filter_type=filter_type)
    assert averaged.shape == random_video.shape


@pytest.mark.parametrize(
        "random_video",
        [
            {"video_shape": (100, 20, 20)}],
        indirect=["random_video"])
@pytest.mark.parametrize(
        "filter_type, call",
        [
            ("uniform", "uniform_filter1d"),
            ("gaussian", "gaussian_filter1d")])
def test_averaging_calls(random_video, filter_type, call, monkeypatch):
    """tests that te filter_typearg specifies the right call
    """
    def mocked_fun_1(video, size, axis, mode):
        return video
    mockfun = MagicMock(side_effect=mocked_fun_1)
    monkeypatch.setattr(averaging, call, mockfun)
    averaging.temporal_filter1d(video=random_video,
                                size=10,
                                filter_type=filter_type)
    mockfun.assert_called_once()
