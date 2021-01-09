import os
import h5py
import numpy as np
import ophys_etl.decrosstalk.active_traces as active_traces


def test_mode_calculation():
    """
    Test the calculation of mode_robust by comparing to known
    results generated from the prototype module.
    """

    seed = 77134
    rng = np.random.RandomState(seed)
    data = rng.normal(8.2, 4.5, size=100)
    assert np.abs(active_traces.mode_robust(data)-3.87178)<0.00001

    seed = 77134
    rng = np.random.RandomState(seed)
    data = rng.poisson(lam=100, size=100)
    assert np.abs(active_traces.mode_robust(data)-98.0)<0.00001

    seed = 77134
    rng = np.random.RandomState(seed)
    data = rng.gamma(20, scale=5, size=100)
    assert np.abs(active_traces.mode_robust(data)-91.40142)<0.00001

    seed = 77134
    rng = np.random.RandomState(seed)
    data = rng.chisquare(19, size=100)
    assert np.abs(active_traces.mode_robust(data)-16.46159)<0.00001


def test_mode_robust_slicing():
    """
    Test that the axis slicing in mode_robust works as expected
    """
    rng = np.random.RandomState(4526)
    data = rng.random_sample((10,10,10))

    # iterate over sliced axes
    for axis in range(3):
        sliced_mode = active_traces.mode_robust(data, axis=axis)

        # make sure all results are distinct (in case
        # rng got in a state where things that should fail
        # actually pass)
        assert len(np.unique(sliced_mode)) == 100

        # iterate over the indices of sliced mode; make sure
        # the results for sliced_mode are equivalent to passing
        # the 1-D slice into mode_robust
        for ii in range(10):
            for jj in range(10):
                if axis == 0:
                    test = data[:,ii,jj]
                elif axis==1:
                    test = data[ii,:,jj]
                elif axis==2:
                    test = data[ii,jj,:]
            assert sliced_mode[ii,jj] == active_traces.mode_robust(test)


def test_evaluate_components():
    """
    Test evaluate_components by comparing to inputs and outputs used
    with prototype
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    data_filename = os.path.join(data_dir, 'evaluate_components_data.h5')
    with h5py.File(data_filename, mode='r') as in_file:
        input_trace = in_file['input_trace'][()]
        for robust in (True, False):
            for event_kernel in (5, 10, 15):
                control_comp = in_file['idx_components_%s_%.2d' % (robust, event_kernel)][()]
                control_fitness = in_file['fitness_%s_%.2d' % (robust, event_kernel)][()]
                control_erfc = in_file['erfc_%s_%.2d' % (robust, event_kernel)][()]
                (test_comp,
                 test_fitness,
                 test_erfc) = active_traces.evaluate_components(input_trace,
                                                                robust_std=robust,
                                                                event_kernel=event_kernel)
                np.testing.assert_array_equal(test_comp, control_comp)
                np.testing.assert_array_equal(test_fitness, control_fitness)
                np.testing.assert_array_equal(test_erfc, control_erfc)


def test_find_event_gaps():
    """
    Test that active_traces.find_event_gaps works as advertisede
    """

    n_neurons = 10
    n_timestamps = 100
    trace_flags = np.zeros((n_neurons, n_timestamps), dtype=bool)

    # several events longer than one
    trace_flags[1,5:12] = True
    trace_flags[1, 34:45] = True
    trace_flags[1, 66:88] = True

    # several events; some of which only encompass one time stamp
    trace_flags[2, 17:19] = True
    trace_flags[2, 30] = True
    trace_flags[2, 70] = True

    # one event at the beginning
    trace_flags[3,0] = True

    # one event at the beginning with finite extent
    trace_flags[4,0:3] = True

    # one event at the end with finite extent
    trace_flags[5, 97:] = True

    # several events, with the last one at the end of the time array
    trace_flags[6, 17:19] = True
    trace_flags[6, 30] = True
    trace_flags[6, 70] = True
    trace_flags[6, 97:] = True

    # several events, with the first one at the beginning
    trace_flags[7, 0:2] = True
    trace_flags[7, 17:19] = True
    trace_flags[7, 30] = True
    trace_flags[7, 70] = True

    # several events, with the first one at the beginning, and one at the end
    trace_flags[8, 0:2] = True
    trace_flags[8, 17:19] = True
    trace_flags[8, 30] = True
    trace_flags[8, 70] = True
    trace_flags[8, 96:] = True

    (event_gaps,
     pre_onset,
     last_event,
     event_gaps_intermediate,
     first_gap,
     last_gap) = active_traces.find_event_gaps(trace_flags)

    # first neuron has no events; verify that all entries are NaNs
    assert np.isnan(event_gaps[0])
    assert np.isnan(pre_onset[0])
    assert np.isnan(last_event[0])
    assert np.isnan(event_gaps_intermediate[0])
    assert np.isnan(first_gap[0])
    assert np.isnan(last_gap[0])

    np.testing.assert_array_equal(event_gaps[1], np.array([5, 22, 21, 12]))
    np.testing.assert_array_equal(pre_onset[1], np.array([33, 65]))
    np.testing.assert_array_equal(last_event[1], np.array([11, 44]))
    np.testing.assert_array_equal(event_gaps_intermediate[1], np.array([22, 21]))
    assert first_gap[1] == [5]
    assert last_gap[1] == [12]

    np.testing.assert_array_equal(event_gaps[2], np.array([17, 11, 39, 29]))
    np.testing.assert_array_equal(pre_onset[2], np.array([29, 69]))
    np.testing.assert_array_equal(last_event[2], np.array([18, 30]))
    np.testing.assert_array_equal(event_gaps_intermediate[2], np.array([11, 39]))
    assert first_gap[2] == [17]
    assert last_gap[2] == [29]

    np.testing.assert_array_equal(event_gaps[3], np.array([99]))
    np.testing.assert_array_equal(pre_onset[3], np.array([]))
    np.testing.assert_array_equal(last_event[3], np.array([]))
    np.testing.assert_array_equal(event_gaps_intermediate[3], np.array([]))
    assert first_gap[3] == []  # is this really what we want?
    assert last_gap[3] == [99]

    np.testing.assert_array_equal(event_gaps[4], np.array([97]))
    np.testing.assert_array_equal(pre_onset[4], np.array([]))
    np.testing.assert_array_equal(last_event[4], np.array([]))
    np.testing.assert_array_equal(event_gaps_intermediate[4], np.array([]))
    assert first_gap[4] == []
    assert last_gap[4] == [97]

    np.testing.assert_array_equal(event_gaps[5], np.array([97]))
    np.testing.assert_array_equal(pre_onset[5], np.array([]))
    np.testing.assert_array_equal(last_event[5], np.array([]))
    np.testing.assert_array_equal(event_gaps_intermediate[5], np.array([]))
    assert first_gap[5] == [97]
    assert last_gap[5] == []

    np.testing.assert_array_equal(event_gaps[6], np.array([17, 11, 39, 26]))
    np.testing.assert_array_equal(pre_onset[6], np.array([29, 69, 96]))
    np.testing.assert_array_equal(last_event[6], np.array([18, 30, 70]))
    np.testing.assert_array_equal(event_gaps_intermediate[6], np.array([11, 39, 26]))
    assert first_gap[6] == [17]
    assert last_gap[6] == []

    np.testing.assert_array_equal(event_gaps[7], np.array([15, 11, 39, 29]))
    np.testing.assert_array_equal(pre_onset[7], np.array([16, 29, 69]))
    np.testing.assert_array_equal(last_event[7], np.array([1, 18, 30]))
    np.testing.assert_array_equal(event_gaps_intermediate[7], np.array([15, 11, 39]))
    assert first_gap[7] == []
    assert last_gap[7] == [29]

    np.testing.assert_array_equal(event_gaps[8], np.array([15, 11, 39, 25]))
    np.testing.assert_array_equal(pre_onset[8], np.array([16, 29, 69, 95]))
    np.testing.assert_array_equal(last_event[8], np.array([1, 18, 30, 70]))
    np.testing.assert_array_equal(event_gaps_intermediate[8], np.array([15, 11, 39, 25]))
    assert first_gap[8] == []
    assert last_gap[8] == []


def test_get_trace_events():
    """
    Test output of get_trace_events against outputs generated with the prototype
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    data_filename = os.path.join(data_dir, 'evaluate_components_data.h5')
    with h5py.File(data_filename, mode='r') as in_file:
        input_trace = in_file['input_trace'][()]
        for len_ne in (10, 20, 30):
            for th_ag in (7, 14, 21):
                suffix = 'len_ne_%.2d_th_ag_%.2d' % (len_ne, th_ag)
                results = active_traces.get_trace_events(input_trace,
                                                         len_ne=len_ne,
                                                         th_ag=th_ag)

                assert len(results['trace']) == in_file['n_trace_%s' % suffix][()]
                assert len(results['trace']) == len(results['events'])
                for i_ev in range(len(results['trace'])):
                    trace_control = in_file['trace_events_%s_%d' % (suffix, i_ev)][()]
                    ind_control = in_file['event_indices_%s_%d' % (suffix, i_ev)][()]
                    np.testing.assert_array_equal(trace_control, results['trace'][i_ev])
                    np.testing.assert_array_equal(ind_control, results['events'][i_ev])
