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
