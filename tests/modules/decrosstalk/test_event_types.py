import pytest
import ophys_etl.modules.decrosstalk.decrosstalk_types as dc_types
import numpy as np
from numpy.testing import assert_array_almost_equal as np_almost
from numpy.testing import assert_array_equal as np_equal


def test_ROIEvents():

    trace = np.linspace(0, 2.0, 5)
    events = np.arange(5, dtype=int)
    roi_events = dc_types.ROIEvents()
    roi_events['trace'] = trace
    roi_events['events'] = events

    np_almost(roi_events['trace'], np.linspace(0, 2.0, 5), decimal=10)
    np_equal(roi_events['events'], np.arange(5, dtype=int))


def test_ROIEvents_exceptions():

    trace = np.linspace(0, 2.0, 5)
    events = np.arange(5, dtype=int)
    roi_events = dc_types.ROIEvents()

    with pytest.raises(KeyError):
        roi_events['boom'] = trace
    with pytest.raises(ValueError):
        roi_events['trace'] = events
    with pytest.raises(ValueError):
        roi_events['events'] = trace
    with pytest.raises(ValueError):
        roi_events['events'] = 5
    with pytest.raises(ValueError):
        roi_events['trace'] = 6.7
    with pytest.raises(ValueError):
        roi_events['trace'] = np.array([[1.1, 2.2], [3.4, 5.5]])

    roi_events['trace'] = trace
    roi_events['events'] = events

    _ = roi_events['trace']
    _ = roi_events['events']
    with pytest.raises(KeyError):
        _ = roi_events['boom']


def test_ROIEventChannels():
    rng = np.random.RandomState(1245)
    traces = list([rng.random_sample(10)
                   for ii in range(2)])
    events = list([rng.randint(0, 20, size=10)
                   for ii in range(2)])

    event_set = dc_types.ROIEventChannels()
    for ii, k in enumerate(('signal', 'crosstalk')):
        ee = dc_types.ROIEvents()
        ee['trace'] = traces[ii]
        ee['events'] = events[ii]
        event_set[k] = ee

    np_almost(event_set['signal']['trace'], traces[0], decimal=10)
    np_equal(event_set['signal']['events'], events[0])
    np_almost(event_set['crosstalk']['trace'], traces[1], decimal=10)
    np_equal(event_set['crosstalk']['events'], events[1])


def test_ROIEventChannels_exceptions():

    event_set = dc_types.ROIEventChannels()
    ee = dc_types.ROIEvents()
    ee['trace'] = np.linspace(0, 1, 7)
    ee['events'] = np.arange(7, dtype=int)

    with pytest.raises(KeyError):
        event_set['a'] = ee

    with pytest.raises(ValueError):
        event_set['signal'] = np.linspace(1, 9, 12)

    event_set['signal'] = ee

    with pytest.raises(KeyError):
        _ = event_set['b']


def test_ROIEventSet():
    event_set = dc_types.ROIEventSet()
    rng = np.random.RandomState(888812)
    true_trace_s = []
    true_event_s = []
    true_trace_c = []
    true_event_c = []
    for ii in range(3):
        t = rng.random_sample(13)
        e = rng.randint(0, 111, size=13)
        signal = dc_types.ROIEvents()
        signal['trace'] = t
        signal['events'] = e
        true_trace_s.append(t)
        true_event_s.append(e)

        t = rng.random_sample(13)
        e = rng.randint(0, 111, size=13)
        crosstalk = dc_types.ROIEvents()
        crosstalk['trace'] = t
        crosstalk['events'] = e
        true_trace_c.append(t)
        true_event_c.append(e)

        channels = dc_types.ROIEventChannels()
        channels['signal'] = signal
        channels['crosstalk'] = crosstalk
        event_set[ii] = channels

    for ii in range(3):
        np_almost(event_set[ii]['signal']['trace'],
                  true_trace_s[ii], decimal=10)
        np_equal(event_set[ii]['signal']['events'],
                 true_event_s[ii])
        np_almost(event_set[ii]['crosstalk']['trace'],
                  true_trace_c[ii], decimal=10)
        np_equal(event_set[ii]['crosstalk']['events'],
                 true_event_c[ii])
    assert 0 in event_set
    assert 1 in event_set
    assert 2 in event_set
    assert 3 not in event_set
    keys = event_set.keys()
    keys.sort()
    assert keys == [0, 1, 2]
    channels = event_set.pop(1)
    np_almost(channels['signal']['trace'],
              true_trace_s[1], decimal=10)
    np_equal(channels['signal']['events'],
             true_event_s[1])
    np_almost(channels['crosstalk']['trace'],
              true_trace_c[1], decimal=10)
    np_equal(channels['crosstalk']['events'],
             true_event_c[1])
    assert 0 in event_set
    assert 2 in event_set
    assert 1 not in event_set
    keys = event_set.keys()
    keys.sort()
    assert keys == [0, 2]


def test_ROIEventSet_exceptions():

    channel = dc_types.ROIEventChannels()
    e = dc_types.ROIEvents()
    e['trace'] = np.linspace(1, 3, 10)
    e['events'] = np.arange(10, dtype=int)
    channel['signal'] = e
    e = dc_types.ROIEvents()
    e['trace'] = np.linspace(1, 7, 10)
    e['events'] = np.arange(10, 20, dtype=int)
    channel['crosstalk'] = e

    event_set = dc_types.ROIEventSet()
    with pytest.raises(KeyError):
        event_set['signal'] = e
    with pytest.raises(ValueError):
        event_set[9] = np.linspace(2, 7, 20)
    event_set[9] = channel
