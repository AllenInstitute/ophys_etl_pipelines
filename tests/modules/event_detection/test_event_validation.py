import pytest
import numpy as np

from ophys_etl.modules.event_detection import validation


@pytest.mark.parametrize(
        "n_samples, index, magnitude, decay_time, rate",
        [
            (100, 50, 12.5, 0.4, 11.0),
            ])
def test_make_event(n_samples, index, magnitude, decay_time, rate):
    trace = validation.make_event(n_samples=n_samples,
                                  index=index,
                                  magnitude=magnitude,
                                  decay_time=decay_time,
                                  rate=rate)
    assert trace.size == n_samples
    assert np.argmax(trace) == index
    assert trace[index] == magnitude
    tau_index_units = decay_time * rate
    np.testing.assert_allclose(trace[index + 1] / trace[index],
                               np.exp(- 1.0 / tau_index_units))
    assert trace[index - 1] == 0.0


@pytest.mark.parametrize(
        "n_samples, timestamps, magnitudes, decay_time, rate",
        [
            (100, [10, 80], [12.5, 4.5], 0.4, 11.0),
            ])
def test_sum_events(n_samples, timestamps, magnitudes, decay_time, rate):
    trace = validation.sum_events(n_samples=n_samples,
                                  timestamps=timestamps,
                                  magnitudes=magnitudes,
                                  decay_time=decay_time,
                                  rate=rate)
    assert trace.size == n_samples
    np.testing.assert_allclose(trace[timestamps], magnitudes, atol=1e-5)
