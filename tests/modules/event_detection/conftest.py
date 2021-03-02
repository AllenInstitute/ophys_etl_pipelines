import pytest
import numpy as np


def make_event(length, index, magnitude, decay_time, rate):
    timestamps = np.arange(length) / rate
    t0 = timestamps[index]
    z = np.zeros(length)
    z[index:] = magnitude * np.exp(-(timestamps[index:] - t0) / decay_time)
    return z


@pytest.fixture
def sum_events_function():
    # returns a function
    def _sum_events_function(nframes, timestamps, magnitudes,
                             decay_time, rate):
        data = np.zeros(nframes)
        for ts, mag in zip(timestamps, magnitudes):
            data += make_event(nframes, ts, mag, decay_time, rate)
        return data
    return _sum_events_function


@pytest.fixture
def sum_events_fixture(request, sum_events_function):
    nframes = request.param.get("nframes")
    timestamps = request.param.get("timestamps")
    magnitudes = request.param.get("magnitudes")
    decay_time = request.param.get("decay_time")
    rate = request.param.get("rate")
    data = sum_events_function(nframes, timestamps, magnitudes,
                               decay_time, rate)
    return data, decay_time, rate
