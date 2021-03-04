import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal as np_almost
import ophys_etl.modules.decrosstalk.decrosstalk_types as dc_types


def test_ROIChannels():

    signal = np.arange(9, dtype=float)
    crosstalk = np.arange(8, 17, dtype=float)
    mm = np.array([[1.2, 3.4], [5.6, 7.9]])
    p_signal = signal+0.01
    p_crosstalk = crosstalk+0.7
    p_mm = mm+0.9
    channels = dc_types.ROIChannels()
    channels['signal'] = signal
    channels['crosstalk'] = crosstalk
    channels['mixing_matrix'] = mm
    channels['poorly_converged_signal'] = p_signal
    channels['poorly_converged_crosstalk'] = p_crosstalk
    channels['poorly_converged_mixing_matrix'] = p_mm
    channels['use_avg_mixing_matrix'] = False

    np_almost(channels['signal'], signal, decimal=10)
    np_almost(channels['crosstalk'], crosstalk, decimal=10)
    np_almost(channels['mixing_matrix'], mm, decimal=10)
    np_almost(p_signal,
              channels['poorly_converged_signal'], decimal=10)
    np_almost(p_crosstalk,
              channels['poorly_converged_crosstalk'], decimal=10)
    np_almost(p_mm,
              channels['poorly_converged_mixing_matrix'], decimal=10)
    assert not channels['use_avg_mixing_matrix']


def test_ROIChannel_exceptions():

    bad_signals = [np.array([1, 2, 3], dtype=int),
                   np.array([[1.1, 2.2], [3.4, 5.4]], dtype=float),
                   1.4]

    for bad_val in bad_signals:
        channels = dc_types.ROIChannels()
        with pytest.raises(ValueError):
            channels['signal'] = bad_val

    for bad_val in bad_signals:
        channels = dc_types.ROIChannels()
        with pytest.raises(ValueError):
            channels['crosstalk'] = bad_val

    for bad_val in bad_signals:
        with pytest.raises(ValueError):
            channels = dc_types.ROIChannels()
            channels['poorly_converged_signal'] = bad_val

    for bad_val in bad_signals:
        channels = dc_types.ROIChannels()
        with pytest.raises(ValueError):
            channels['poorly_converged_crosstalk'] = bad_val

    bad_mm = [np.array([1.5, 6.7, 3.4]),
              np.array([[1, 3], [7, 9]], dtype=int),
              4.5]
    for bad_val in bad_mm:
        channels = dc_types.ROIChannels()
        with pytest.raises(ValueError):
            channels['mixing_matrix'] = bad_val
    for bad_val in bad_mm:
        channels = dc_types.ROIChannels()
        with pytest.raises(ValueError):
            channels['poorly_converged_mixing_matrix'] = bad_val

    channels = dc_types.ROIChannels()
    with pytest.raises(ValueError):
        channels['use_avg_mixing_matrix'] = 'True'

    channels = dc_types.ROIChannels()
    channels['signal'] = np.arange(9, dtype=float)
    with pytest.raises(NotImplementedError):
        channels.pop('signal')

    channels = dc_types.ROIChannels()
    with pytest.raises(KeyError):
        channels['abracadabra'] = 9


def test_ROIChannels_in():

    signal = np.arange(9, dtype=float)
    crosstalk = np.arange(8, 17, dtype=float)
    mm = np.array([[1.2, 3.4], [5.6, 7.9]])
    p_signal = signal+0.01
    p_crosstalk = crosstalk+0.7
    p_mm = mm+0.9
    channels = dc_types.ROIChannels()
    channels['signal'] = signal
    assert 'signal' in channels
    assert 'crosstalk' not in channels
    assert 'mixing_matrix' not in channels
    assert 'poorly_converged_signal' not in channels
    assert 'poorly_converged_crosstalk' not in channels
    assert 'poorly_converged_mixing_matrix' not in channels
    assert 'use_avg_mixing_matrix' not in channels

    channels['crosstalk'] = crosstalk
    assert 'signal' in channels
    assert 'crosstalk' in channels
    assert 'mixing_matrix' not in channels
    assert 'poorly_converged_signal' not in channels
    assert 'poorly_converged_crosstalk' not in channels
    assert 'poorly_converged_mixing_matrix' not in channels
    assert 'use_avg_mixing_matrix' not in channels

    channels['mixing_matrix'] = mm
    assert 'signal' in channels
    assert 'crosstalk' in channels
    assert 'mixing_matrix' in channels
    assert 'poorly_converged_signal' not in channels
    assert 'poorly_converged_crosstalk' not in channels
    assert 'poorly_converged_mixing_matrix' not in channels
    assert 'use_avg_mixing_matrix' not in channels

    keys = channels.keys()
    keys.sort()
    assert keys == ['crosstalk', 'mixing_matrix', 'signal']

    channels['poorly_converged_signal'] = p_signal
    assert 'signal' in channels
    assert 'crosstalk' in channels
    assert 'mixing_matrix' in channels
    assert 'poorly_converged_signal' in channels
    assert 'poorly_converged_crosstalk' not in channels
    assert 'poorly_converged_mixing_matrix' not in channels
    assert 'use_avg_mixing_matrix' not in channels

    channels['poorly_converged_crosstalk'] = p_crosstalk
    assert 'signal' in channels
    assert 'crosstalk' in channels
    assert 'mixing_matrix' in channels
    assert 'poorly_converged_signal' in channels
    assert 'poorly_converged_crosstalk' in channels
    assert 'poorly_converged_mixing_matrix' not in channels
    assert 'use_avg_mixing_matrix' not in channels

    keys = channels.keys()
    keys.sort()
    assert keys == ['crosstalk', 'mixing_matrix',
                    'poorly_converged_crosstalk',
                    'poorly_converged_signal',
                    'signal']

    channels['poorly_converged_mixing_matrix'] = p_mm
    assert 'signal' in channels
    assert 'crosstalk' in channels
    assert 'mixing_matrix' in channels
    assert 'poorly_converged_signal' in channels
    assert 'poorly_converged_crosstalk' in channels
    assert 'poorly_converged_mixing_matrix' in channels
    assert 'use_avg_mixing_matrix' not in channels

    channels['use_avg_mixing_matrix'] = False
    assert 'signal' in channels
    assert 'crosstalk' in channels
    assert 'mixing_matrix' in channels
    assert 'poorly_converged_signal' in channels
    assert 'poorly_converged_crosstalk' in channels
    assert 'poorly_converged_mixing_matrix' in channels
    assert 'use_avg_mixing_matrix' in channels

    keys = channels.keys()
    keys.sort()
    assert keys == ['crosstalk',
                    'mixing_matrix',
                    'poorly_converged_crosstalk',
                    'poorly_converged_mixing_matrix',
                    'poorly_converged_signal',
                    'signal',
                    'use_avg_mixing_matrix']


def test_ROIDict():

    channel = dc_types.ROIChannels()
    channel['signal'] = np.array([9.2, 3.4, 6.7])
    channel['use_avg_mixing_matrix'] = False
    roi_dict = dc_types.ROIDict()
    roi_dict[9] = channel

    np_almost(roi_dict[9]['signal'],
              np.array([9.2, 3.4, 6.7]),
              decimal=9)

    assert not roi_dict[9]['use_avg_mixing_matrix']


def test_ROIDict_exceptions():

    channel = dc_types.ROIChannels()
    channel['signal'] = np.array([9.2, 3.4, 6.7])
    channel['use_avg_mixing_matrix'] = False

    roi_dict = dc_types.ROIDict()
    with pytest.raises(KeyError):
        roi_dict['aa'] = channel

    roi_dict = dc_types.ROIDict()
    with pytest.raises(ValueError):
        roi_dict[11] = 'abcde'


def test_ROIDict_pop_and_keys():
    rng = np.random.RandomState(1123)
    s1 = rng.random_sample(10)
    c1 = rng.random_sample(10)
    s2 = rng.random_sample(14)
    c2 = rng.random_sample(14)

    channel1 = dc_types.ROIChannels()
    channel1['signal'] = s1
    channel1['crosstalk'] = c1

    channel2 = dc_types.ROIChannels()
    channel2['signal'] = s2
    channel2['crosstalk'] = c2

    roi_dict = dc_types.ROIDict()
    roi_dict[88] = channel1
    roi_dict[77] = channel2

    keys = roi_dict.keys()
    keys.sort()
    assert keys == [77, 88]

    assert 77 in roi_dict
    assert 88 in roi_dict
    assert 55 not in roi_dict

    test = roi_dict.pop(88)
    assert 77 in roi_dict
    assert 88 not in roi_dict
    assert roi_dict.keys() == [77]

    np_almost(test['signal'], s1, decimal=10)
    np_almost(test['crosstalk'], c1, decimal=10)

    np_almost(roi_dict[77]['signal'], s2, decimal=10)
    np_almost(roi_dict[77]['crosstalk'], c2, decimal=10)


def test_ROISetDict():

    rng = np.random.RandomState(53124)
    signals = list([rng.random_sample(10) for ii in range(4)])
    crosstalks = list([rng.random_sample(10) for ii in range(4)])

    roi_set_dict = dc_types.ROISetDict()

    for ii in range(2):
        c = dc_types.ROIChannels()
        c['signal'] = signals[ii]
        c['crosstalk'] = crosstalks[ii]
        roi_set_dict['roi'][ii] = c

    for ii in range(2, 4):
        c = dc_types.ROIChannels()
        c['signal'] = signals[ii]
        c['crosstalk'] = crosstalks[ii]
        roi_set_dict['neuropil'][ii-2] = c

    np_almost(roi_set_dict['roi'][0]['signal'],
              signals[0], decimal=10)

    np_almost(roi_set_dict['roi'][1]['signal'],
              signals[1], decimal=10)

    np_almost(roi_set_dict['roi'][0]['crosstalk'],
              crosstalks[0], decimal=10)

    np_almost(roi_set_dict['roi'][1]['crosstalk'],
              crosstalks[1], decimal=10)

    np_almost(roi_set_dict['neuropil'][0]['signal'],
              signals[2], decimal=10)

    np_almost(roi_set_dict['neuropil'][1]['signal'],
              signals[3], decimal=10)

    np_almost(roi_set_dict['neuropil'][0]['crosstalk'],
              crosstalks[2], decimal=10)

    np_almost(roi_set_dict['neuropil'][1]['crosstalk'],
              crosstalks[3], decimal=10)


def test_ROISetDict_exceptions():

    roi_set = dc_types.ROISetDict()
    with pytest.raises(NotImplementedError):
        roi_set['roi'] = dc_types.ROIDict()

    with pytest.raises(KeyError):
        roi_set[9]
