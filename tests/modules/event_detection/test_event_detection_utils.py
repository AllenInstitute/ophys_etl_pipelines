import pytest
import numpy as np
from ophys_etl.modules.event_detection import utils


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        "sum_events_fixture",
        [
            {
                "nframes": 1000,
                "timestamps": [45, 112, 232, 410, 490, 700, 850],
                "magnitudes": [4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0],
                "decay_time": 0.4,
                "rate": 11.0},
            {
                "nframes": 1000,
                "timestamps": [45, 112, 232, 410, 490, 700, 850],
                "magnitudes": [4.0, 5.0, 6.0, 5.0, 5.5, 5.0, 7.0],
                "decay_time": 0.4,
                "rate": 31.0}], indirect=['sum_events_fixture'])
def test_fast_lzero(sum_events_fixture):
    data, decay_time, rate = sum_events_fixture
    halflife = utils.calculate_halflife(decay_time)
    gamma = utils.calculate_gamma(halflife, rate)
    f = utils.fast_lzero(1.0, data, gamma, True)
    assert f.shape == data.shape


@pytest.mark.event_detect_only
@pytest.mark.parametrize(
        "frac_outliers, threshold",
        [
            (0.025, 0.1),
            (0.05, 0.1),
            ])
def test_trace_noise_estimate(frac_outliers, threshold):
    """makes a low-frequency signal with noise and outliers and
    checks that the trace noise estimate meets some threshold
    """
    filt_length = 31
    npts = 10000
    sigma = 1.0
    rng = np.random.default_rng(42)
    x = 0.2 * np.cos(2.0 * np.pi * np.arange(npts) / (filt_length * 10))
    x += rng.standard_normal(npts) * sigma
    inds = rng.integers(0, npts, size=int(frac_outliers * npts))
    x[inds] *= 100
    rstd = utils.trace_noise_estimate(x, filt_length)
    assert np.abs(rstd - sigma) < threshold
