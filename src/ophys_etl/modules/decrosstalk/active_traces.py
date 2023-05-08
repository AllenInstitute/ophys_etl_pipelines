from typing import Union, Tuple, List, Dict, Optional
import numpy as np
import scipy.stats

__all__ = ['get_trace_events',
           'evaluate_components',
           'mode_robust']


def mode_robust(input_data: np.ndarray,
                axis: Optional[int] = None) -> Union[np.ndarray,
                                                     int,
                                                     float]:
    """
    Robust estimator of the mode of a data set using the half-sample mode.

    .. versionadded: 1.0.3

    Parameters
    ----------
    input_data -- a np.ndarray of data whose mode is desired

    axis -- int; the axis along which to take the mode

    Returns
    -------
    A scalar (or, if slicing along an axis, an np.ndarray)
    containing the mode of input_data
    """
    if axis is not None:
        def fnc(x: np.ndarray): return mode_robust(x)
        data_mode = np.apply_along_axis(fnc, axis, input_data)
    else:
        # Create the function that we can use for the half-sample mode
        def _hsm(dt_input: np.ndarray) -> Union[int, float, np.ndarray]:
            # the presence of NaNs fouls up the algorithm
            # just replace them with a nonsense vlaue
            dt = np.where(np.isnan(dt_input), -999., dt_input)

            if dt.size == 1:
                return dt[0]
            elif dt.size == 2:
                return dt.mean()
            elif dt.size == 3:
                i1 = dt[1] - dt[0]
                i2 = dt[2] - dt[1]
                if i1 < i2:
                    return dt[:2].mean()
                elif i2 > i1:
                    return dt[1:].mean()
                else:
                    return dt[1]
            else:

                w_min = np.inf
                n = dt.size / 2 + dt.size % 2
                n = int(n)
                for i in range(0, n):
                    w = dt[i + n - 1] - dt[i]
                    if w < w_min:
                        w_min = w
                        j = i

                return _hsm(dt[j:j + n])

        data = input_data.ravel()  # flatten all dimensions
        if type(data).__name__ == "MaskedArray":
            data = data.compressed()

        # The data need to be sorted for this to work
        data = np.sort(data)

        # Find the mode
        data_mode = _hsm(data)

    return data_mode


def evaluate_components(traces: np.ndarray,
                        event_kernel: int = 5,
                        robust_std: bool = False) -> Tuple[np.ndarray,
                                                           np.ndarray,
                                                           np.ndarray]:
    """
    Define a metric and order components according to the
    probability if some "exceptional events" (like a spike).
    Such probability is defined as the likelihood of
    observing the actual trace value over N samples given
    an estimated noise distribution. The function first
    estimates the noise distribution by considering the
    dispersion around the mode. This is done only using values
    lower than the mode. The estimation of the noise std is
    made robust by using the approximation std=iqr/1.349. Then,
    the probability of having N consecutive events is estimated.
    This probability is used to order the components.

    Created on Tue Aug 23 09:40:37 2016
    @author: Andrea G with small modifications from farznaj

    :param traces: numpy.array of size NxM where
                          N : number of neurons (rois),
                          M: number of timestamps (Fluorescence traces)

    :param event_kernel: int, number of consecutive events

    :param robust_std: boolean indicating whether or not
                       to use the interquartile
                       range to calcluate standard deviation

    :return

        idx_components: numpy.array;
                        the components ordered according to the fitness

        fitness: numpy.array;

        erfc: numpy.array;
              probability at each time step of observing the
              N consecutive actual trace values given the distribution
              of noise. For each event, this is the sum of the log
              probabilities of finding a series of {event_kernel} events
              at least as extreme as the {event_kernel} sequence of
              events starting with designated event.

    """

    n_timesteps = np.shape(traces)[-1]
    mode = mode_robust(traces, axis=1)
    ff1 = traces - mode[:, None]

    # only consider values under the mode
    # to determine the noise standard deviation
    ff1 = -ff1 * (ff1 < 0)

    if robust_std:
        # compute 25 percentile
        ff1 = np.sort(ff1, axis=1)
        ff1[ff1 == 0] = np.nan
        ns = np.round(np.sum(ff1 > 0, 1) * .5).astype(int)
        iqr_h = np.zeros(traces.shape[0])

        for idx, el in enumerate(ff1):
            iqr_h[idx] = ff1[idx, -ns[idx]]

        # approximate standard deviation as iqr/1.349
        sd_r = 2 * iqr_h / 1.349

    else:
        ns = np.sum(ff1 > 0, 1)
        sd_r = np.sqrt(np.sum(ff1 ** 2, 1) / ns)

    # compute z value
    z = (traces - mode[:, None]) / (3 * sd_r[:, None])

    # probability of observing values larger or equal to z given normal
    # distribution with mean=mode and std sd_r
    erf = 1 - scipy.stats.norm.cdf(z)

    # use logarithm so that multiplication becomes sum
    # np.errstate is to suppress "Divide by zero" warning from np.log
    with np.errstate(divide='ignore'):
        erf = np.log(erf)
        # ToDo: Debug "RuntimeWarning: divide by zero encountered in log"

    # build kernel for event detection
    filt = np.ones(event_kernel)

    # convolve probability with kernel
    erfc = np.apply_along_axis(lambda m: np.convolve(
        m, filt, mode='full'), axis=1, arr=erf)
    erfc = erfc[:, :n_timesteps]

    # select the maximum value of such probability for each trace
    fitness = np.min(erfc, 1)

    ordered = np.argsort(fitness)

    idx_components = ordered  # [::-1]# selec only portion of components

    #    fitness = fitness[idx_components] % FN
    #    commented bc we want the indexing to match C and YrA.

    #    erfc = erfc[idx_components] % FN
    #    commented bc we want the indexing to match C and YrA.

    return idx_components, fitness, erfc


def _trace_to_flag(traces_y0: np.ndarray,
                   th_ag: float) -> np.ndarray:
    """
    Take traces_y0 (np.array of trace values for each neuron) and th_ag;
    return array of boolean flags indicating which timesteps are a part
    of events and which are not

    Parameters
    ----------
    traces_y0 -- an MxN numpy array containing the traces of
    the neurons (M is the number of neurons; N is the number
    of time steps)

    th_ag -- a threshold applied to the negative sum of log
    probabilities of each series of 5 events. Any series
    of 5 events whose sum of log probabilities exceeds this
    parameter indicates the onset of an "event"

    Returns
    -------
    a MxN numpy array of booleans indicating which elements in
    traces_y0 were active events
    """

    #  Andrea Giovannucci's method of identifying "exceptional" events
    [_, _, erfc] = evaluate_components(traces_y0,
                                       event_kernel=5,
                                       robust_std=False)
    erfc = -erfc

    # applying threshold
    evs = (erfc >= th_ag)  # neurons x frames

    return evs


def _flag_to_events(traces_y0: np.ndarray,
                    evs: np.ndarray,
                    len_ne: int) -> Tuple[List[np.ndarray],
                                          List[np.ndarray]]:
    """
    Take an array of traces and an array of booleans marking
    "active" time steps in those traces. Return, for each neuron,
    numpy arrays containing only the active timesteps.

    Parameters
    ----------
    traces_y0 -- an MxN numpy array containing the traces of
    the neurons (M is the number of neurons; N is the number
    of time steps)

    evs -- an MxN numpy array of booleans indicating which
    elements of traces_y0 are "active"

    len_ne -- an int; the number of time steps before and
    after each event to include the in output trace

    Returns
    -------
    traces_out -- a list of numpy.arrays. Each array
    contains the active trace elements for the corresponding
    neuron.

    events_out -- a list of numpy.arrays. Each array is
    the indices of the active elements from traces_out,

    i.e.

    traces_y0[2, events_out[2]] == traces_out[2]
    """

    n_neurons = traces_y0.shape[0]
    n_time = traces_y0.shape[1]

    # initialize arrays of indices for finding
    # the len_ne pixels to either side of events
    # in trace
    index_minus = {}
    index_plus = {}
    for dx in range(1, len_ne+1, 1):
        _plus = np.arange(n_time-dx, dtype=int)
        _minus = np.arange(dx, n_time, dtype=int)
        assert _minus.shape == _plus.shape
        index_minus[dx] = _minus
        index_plus[dx] = _plus

    # adjust evs so that the timesteps
    # that are len_ne to either side of "natural"
    # events are also marked as events
    new_evs = np.copy(evs)
    for i_neuron in range(n_neurons):
        active_mask = new_evs[i_neuron, :]

        # get len_ne to either side of every event
        extra_indices = []
        for dx in range(1, len_ne+1, 1):
            _minus = index_minus[dx]
            _plus = index_plus[dx]
            valid_minus = np.where(active_mask[_minus])[0]
            valid_minus = _minus[valid_minus] - dx

            valid_plus = np.where(active_mask[_plus])[0]
            valid_plus = _plus[valid_plus]+dx

            extra_indices.append(valid_plus)
            extra_indices.append(valid_minus)
        extra_indices = np.unique(np.concatenate(extra_indices))
        new_evs[i_neuron, extra_indices] = True

    traces_out = []
    events_out = []

    for i_neuron in range(n_neurons):
        _traces = traces_y0[i_neuron, new_evs[i_neuron, :]]
        _events = np.where(new_evs[i_neuron, :])[0]

        traces_out.append(_traces.astype(float))
        events_out.append(_events.astype(int))

    return (traces_out, events_out)


def get_traces_evs(traces_y0: np.ndarray,
                   th_ag: float,
                   len_ne: int) -> Dict[str, np.ndarray]:
    """
    Function to get an "active trace" i.e. a trace made by extracting
    and concatenating the active parts of the input trace

    Farzaneh Najafi
    March 2020

    :param traces_y0: numpy.array of size NxM
                      N : number of neurons (rois), M: number of timestamps

    :param th_ag: scalar : threshold to find events on the trace;
                           the higher the more strict on what we call an event.
                           This threshold is applied to the negative of the
                           {erfc} array returned by evaluate_components with
                           {event_kernel}=5 (so, it is a threshold applied on
                           a sum of log probabilities)

    :param len_ne: scalar; number of frames before and after each event that
                           are taken to create traces_events

    :return:
        a dict with keys
        'trace': ndarray, size N (number of neurons);
                          each neuron has size n, which is the size of the
                          "active trace" for that neuron. For each neuron,
                          this array contains the trace values at the active
                          events in that trace.

        'events': ndarray, size number_of_neurons;
                  indices to apply on traces_y0 to get traces_y0_evs:
    """

    evs = _trace_to_flag(traces_y0, th_ag)
    traces, inds = _flag_to_events(traces_y0, evs, len_ne)

    return {'trace': traces,
            'events': inds}


def get_trace_events(traces: np.ndarray,
                     threshold_parameters: Dict[str, Union[int, float]]
                     ) -> Dict[str, np.ndarray]:
    """
    Wrapper around event detection code.

    Params:
    -------
    traces -- an NxM numpy array. N is the number of ROIs. M is the number
    of timesteps. Contains the trace flux values

    threshold_parameters -- a dict of event detection parameters that
    will be passed to the the actual event detection code

    Returns
    -------
    a dict with keys
        'trace': ndarray, size N (number of neurons);
                          each neuron has size n, which is the size of the
                          "active trace" for that neuron. For each neuron,
                          this array contains the trace values at the active
                          events in that trace.

        'events': ndarray, size number_of_neurons;
                  indices to apply on traces_y0 to get traces_y0_evs:

    """

    return get_traces_evs(traces, **threshold_parameters)
