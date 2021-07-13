import pytest

from ophys_etl.modules.event_detection import fast_lzero_utils, utils
from ophys_etl.modules.event_detection import validation


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        "nframes, timestamps, magnitudes, decay_time, rate",
        [
            (
                1000,
                [45, 112, 232, 410, 490, 700, 850],
                [4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0],
                0.4,
                11.0),
            (
                1000,
                [45, 112, 232, 410, 490, 700, 850],
                [4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0],
                0.4,
                31.0)])
def test_fast_lzero(nframes, timestamps, magnitudes, decay_time, rate):
    data = validation.sum_events(nframes, timestamps, magnitudes,
                                 decay_time, rate)
    halflife = utils.calculate_halflife(decay_time)
    gamma = utils.calculate_gamma(halflife, rate)
    f = fast_lzero_utils.fast_lzero(1.0, data, gamma, True)
    assert f.shape == data.shape
