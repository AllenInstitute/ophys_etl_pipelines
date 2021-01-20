import os
import h5py
import numpy as np
from numpy.testing import assert_array_almost_equal as np_almost_equal
from numpy.testing import assert_array_equal as np_equal
import ophys_etl.decrosstalk.active_traces as at

from .utils import get_data_dir


def test_mode_calculation():
    """
    Test the calculation of mode_robust by comparing to known
    results generated from the prototype module.
    """

    seed = 77134
    rng = np.random.RandomState(seed)
    data = rng.normal(8.2, 4.5, size=100)
    assert np.abs(at.mode_robust(data)-3.87178) < 0.00001

    seed = 77134
    rng = np.random.RandomState(seed)
    data = rng.poisson(lam=100, size=100)
    assert np.abs(at.mode_robust(data)-98.0) < 0.00001

    seed = 77134
    rng = np.random.RandomState(seed)
    data = rng.gamma(20, scale=5, size=100)
    assert np.abs(at.mode_robust(data)-91.40142) < 0.00001

    seed = 77134
    rng = np.random.RandomState(seed)
    data = rng.chisquare(19, size=100)
    assert np.abs(at.mode_robust(data)-16.46159) < 0.00001


def test_mode_robust_slicing():
    """
    Test that the axis slicing in mode_robust works as expected
    """
    rng = np.random.RandomState(4526)
    data = rng.random_sample((10, 10, 10))

    # iterate over sliced axes
    for axis in range(3):
        sliced_mode = at.mode_robust(data, axis=axis)

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
                    test = data[:, ii, jj]
                elif axis == 1:
                    test = data[ii, :, jj]
                elif axis == 2:
                    test = data[ii, jj, :]
            assert sliced_mode[ii, jj] == at.mode_robust(test)


def test_evaluate_components():
    """
    Test evaluate_components by comparing to inputs and outputs used
    with prototype
    """
    data_dir = get_data_dir()
    data_filename = os.path.join(data_dir, 'evaluate_components_data.h5')
    with h5py.File(data_filename, mode='r') as in_file:
        input_trace = in_file['input_trace'][()]
        for robust in (True, False):
            for event_kernel in (5, 10, 15):
                control_comp = in_file['idx_components_%s_%.2d' %
                                       (robust, event_kernel)][()]

                control_fitness = in_file['fitness_%s_%.2d' %
                                          (robust, event_kernel)][()]

                control_erfc = in_file['erfc_%s_%.2d' %
                                       (robust, event_kernel)][()]
                (test_comp,
                 test_fitness,
                 test_erfc) = at.evaluate_components(input_trace,
                                                     robust_std=robust,
                                                     event_kernel=event_kernel)
                np_almost_equal(test_comp, control_comp, decimal=10)
                np_almost_equal(test_fitness, control_fitness, decimal=10)
                np_almost_equal(test_erfc, control_erfc, decimal=10)


def test_flag_to_events():

    rng = np.random.RandomState(77123)

    n_t = 30
    input_events = []
    input_traces = []
    output_events = []

    len_ne = 2

    ff = np.zeros(n_t, dtype=bool)
    ff[:10] = True
    tt = rng.random_sample(n_t)
    input_events.append(ff)
    input_traces.append(tt)
    output_events.append(np.arange(12, dtype=int))

    ff = np.zeros(n_t, dtype=bool)
    ff[:10] = True
    ff[18:22] = True
    tt = rng.random_sample(n_t)
    input_events.append(ff)
    input_traces.append(tt)
    output_events.append(np.concatenate([np.arange(12, dtype=int),
                                         np.arange(16, 24, dtype=int)]))

    ff = np.zeros(n_t, dtype=bool)
    ff[:10] = True
    ff[18:22] = True
    ff[27:] = True
    tt = rng.random_sample(n_t)
    input_events.append(ff)
    input_traces.append(tt)
    output_events.append(np.concatenate([np.arange(12, dtype=int),
                                         np.arange(16, 24, dtype=int),
                                         np.arange(25, n_t, dtype=int)]))

    ff = np.zeros(n_t, dtype=bool)
    ff[5:10] = True
    ff[18:22] = True
    ff[27:] = True
    tt = rng.random_sample(n_t)
    input_events.append(ff)
    input_traces.append(tt)
    output_events.append(np.concatenate([np.arange(3, 12, dtype=int),
                                         np.arange(16, 24, dtype=int),
                                         np.arange(25, n_t, dtype=int)]))

    ff = np.zeros(n_t, dtype=bool)
    ff[5:10] = True
    ff[18:21] = True
    ff[26] = True
    tt = rng.random_sample(n_t)
    input_events.append(ff)
    input_traces.append(tt)
    output_events.append(np.concatenate([np.arange(3, 12, dtype=int),
                                         np.arange(16, 23, dtype=int),
                                         np.arange(24, 29, dtype=int)]))

    ff = np.zeros(n_t, dtype=bool)
    ff[5:10] = True
    ff[18:21] = True
    ff[23:25] = True
    tt = rng.random_sample(n_t)
    input_events.append(ff)
    input_traces.append(tt)
    output_events.append(np.concatenate([np.arange(3, 12, dtype=int),
                                         np.arange(16, 27, dtype=int)]))

    ff = np.zeros(n_t, dtype=bool)
    ff[:10] = True
    ff[12:15] = True
    ff[23:25] = True
    tt = rng.random_sample(n_t)
    input_events.append(ff)
    input_traces.append(tt)
    output_events.append(np.concatenate([np.arange(17, dtype=int),
                                         np.arange(21, 27, dtype=int)]))

    ff = np.zeros(n_t, dtype=bool)
    ff[5:10] = True
    ff[18:21] = True
    ff[23:] = True
    tt = rng.random_sample(n_t)
    input_events.append(ff)
    input_traces.append(tt)
    output_events.append(np.concatenate([np.arange(3, 12, dtype=int),
                                         np.arange(16, n_t, dtype=int)]))

    input_traces = np.array(input_traces)
    input_events = np.array(input_events)

    (test_traces,
     test_events) = at._flag_to_events(input_traces,
                                       input_events,
                                       len_ne=len_ne)

    for ii in range(len(input_traces)):
        np_equal(test_events[ii], output_events[ii])
        np_almost_equal(test_traces[ii],
                        input_traces[ii][output_events[ii]],
                        decimal=10)
