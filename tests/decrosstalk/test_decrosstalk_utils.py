import numpy as np
import pytest
import ophys_etl.decrosstalk.decrosstalk_utils as decrosstalk_utils

@pytest.mark.parametrize("sig_indices,ct_indices,window,ind_indices",
                         [(np.array([0,10,11,12,21,22,23,24,25,30,31]),
                           np.array([1,9,10,25,26,27,28,32]),
                           1,
                           np.array([3,4,5,6,9])),
                          (np.array([1,10,11,12,21,22,23,24,25,30,31]),
                           np.array([0,9,13,14,15,19,20,21,30]),
                           1,
                           np.array([2,6,7,8])),
                          (np.array([0,10,11,12,21,22,23,24,25,30,31]),
                           np.array([2,8,9,26,27,28]),
                           2,
                           np.array([3,4,5,6,10])),
                          (np.array([2,10,11,12,21,22,23,24,25,30,31]),
                           np.array([0,8,9,26,27,33]),
                           2,
                           np.array([3,4,5,6,9]))
                          ])
def test_find_independent_events(sig_indices,
                                 ct_indices,
                                 window,
                                 ind_indices):
    """
    Test that find_independent_events works by constructing dummy datasets and
    running them through the method

    Arguments
    ---------
    sig_indices -- the contents of signal_events['events']

    ct_indices -- the contents of crosstalk_events['events']

    window -- the width of the blurring window in find_independent events

    ind_indices -- an array such that signal_events['events'][ind_indices]
                   gives the expected output of indpendent_events['events']
    """

    signal_events = {}
    signal_events['events'] = sig_indices
    signal_events['trace'] = np.power(1.1, signal_events['events'])

    # just to make sure we aren't accidentally setting ourselves up to miss
    # a failure due to some degeneracy in the input
    assert len(np.unique(signal_events['trace'])) == len(signal_events['trace'])

    crosstalk_events = {}
    crosstalk_events['events'] = ct_indices

    ind_events = decrosstalk_utils.find_independent_events(signal_events,
                                                           crosstalk_events,
                                                           window=window)

    np.testing.assert_array_equal(ind_events['events'],
                                  signal_events['events'][ind_indices])

    np.testing.assert_array_equal(ind_events['trace'],
                                  signal_events['trace'][ind_indices])

@pytest.mark.parametrize("sig_indices,ct_indices,window,ind_indices",
                         [(np.array([0,10,11,12,21,22,23,24,25,30,31]),
                           np.array([1,9,10,25,26,27,28,32]),
                           1,
                           np.array([3,4,5,6,9])),
                          (np.array([1,10,11,12,21,22,23,24,25,30,31]),
                           np.array([0,9,13,14,15,19,20,21,30]),
                           1,
                           np.array([2,6,7,8])),
                          (np.array([0,10,11,12,21,22,23,24,25,30,31]),
                           np.array([2,8,9,26,27,28]),
                           2,
                           np.array([3,4,5,6,10])),
                          (np.array([2,10,11,12,21,22,23,24,25,30,31]),
                           np.array([0,8,9,26,27,33]),
                           2,
                           np.array([3,4,5,6,9])),
                          (np.array([2,10,11,12,21,22,23,24,25,30,31]),
                           np.array([0,8,9,10,23,24,25,26,27,28,29]),
                           2,
                           np.array([])),
                          (np.array([2,10,11,12,21,22,23,24,25,30,31]),
                           np.array([3,11,12,13,20,21,22,23,26,30]),
                           1,
                           np.array([])),
                          ])
def test_validate_cell_crosstalk(sig_indices,
                                 ct_indices,
                                 window,
                                 ind_indices):
    """
    Test that validate_cell_crosstalk works by constructing dummy datasets and
    running them through the method

    Arguments
    ---------
    sig_indices -- the contents of signal_events['events']

    ct_indices -- the contents of crosstalk_events['events']

    window -- the width of the blurring window in find_independent events

    ind_indices -- an array such that signal_events['events'][ind_indices]
                   gives the expected output of indpendent_events['events']
    """

    signal_events = {}
    signal_events['events'] = sig_indices
    signal_events['trace'] = np.power(1.1, signal_events['events'])

    # just to make sure we aren't accidentally setting ourselves up to miss
    # a failure due to some degeneracy in the input
    assert len(np.unique(signal_events['trace'])) == len(signal_events['trace'])

    crosstalk_events = {}
    crosstalk_events['events'] = ct_indices

    (is_valid_roi,
        ind_events) = decrosstalk_utils.validate_cell_crosstalk(signal_events,
                                                                crosstalk_events,
                                                                window=window)

    if len(ind_indices)>0:
        assert is_valid_roi

        np.testing.assert_array_equal(ind_events['events'],
                                      signal_events['events'][ind_indices])

        np.testing.assert_array_equal(ind_events['trace'],
                                      signal_events['trace'][ind_indices])

    else:
        assert not is_valid_roi
        assert len(ind_events['trace']) == 0
        assert len(ind_events['events']) == 0

def test_validate_raw_traces():

    t1 = np.arange(10).astype(float)
    t2 = np.arange(20).astype(float)

    traces_dict = {}
    traces_dict['roi'] = {}
    traces_dict['neuropil'] = {}

    for roi_id in range(9):
        traces_dict['roi'][roi_id] = {}
        traces_dict['neuropil'][roi_id] = {}
        for k in ('signal', 'crosstalk'):
            traces_dict['roi'][roi_id][k] = np.copy(t1)
            traces_dict['neuropil'][roi_id][k] = np.copy(t1)

    traces_dict['roi'][1]['signal'] = np.copy(t2)
    traces_dict['roi'][2]['crosstalk'] = np.copy(t2)
    traces_dict['neuropil'][3]['signal'] = np.copy(t2)
    traces_dict['neuropil'][4]['crosstalk'] = np.copy(t2)

    traces_dict['roi'][5]['signal'][4] = np.NaN
    traces_dict['roi'][6]['crosstalk'][4] = np.NaN
    traces_dict['neuropil'][7]['signal'][4] = np.NaN
    traces_dict['neuropil'][8]['crosstalk'][4] = np.NaN

    results = decrosstalk_utils.validate_traces(traces_dict)
    assert results == {0:True, 1:False, 2:False, 3:False,
                       4:False, 5:False, 6:False, 7:False,
                       8:False}
