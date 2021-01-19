import numpy as np
import scipy.stats


def get_crosstalk_data(signal, crosstalk):
    """
    Use linear regression to calculate the ratio between signal
    and crosstalk in an ROI.

    Parameters
    ----------
    signal -- np.array of signal trace
    crosstalk -- np.array of crosstalk trace

    Returns
    -------
    A dict
        {'slope': the linear regression slope relating signal to crosstalk,
         'offset': the offset form linear regression,
         'r_value': the Pearson's R-coefficient from linear regression}
    """
    result = scipy.stats.linregress(signal, crosstalk)

    return {'slope': result.slope,
            'offset': result.intercept,
            'r_value': result.rvalue}


def validate_traces(trace_dict):
    """
    Check a traces_dict for validity.
    Validity is defined as neuropile and roi traces having the same shape.
    No NaNs appearing in any trace.

    Parameters
    -----------
    trace_dict contains the traces to be validated. It has a structure like:

        trace_dict['roi'][roi_id]['signal'] = np.array of trace of signal
                                              values for ROI

        trace_dict['roi'][roi_id]['crosstalk'] = np.array of trace of
                                                 crosstalk values for ROI

        trace_dict['neuropil'][roi_id]['signal'] = np.array of trace of
                                                   signal values defined
                                                   in the neuropil
                                                   around ROI

        trace_dict['neuropil'][roi_id]['crosstalk'] = np.array of trace of
                                                      crosstalk values
                                                      defined in the
                                                      neuropil around ROI

    Returns
    -------
    A dict mapping roi_id to a boolean indicating whether or not the trace
    was valid.
    """
    output_dict = {}
    for roi_id in trace_dict['roi'].keys():
        roi_s_trace = trace_dict['roi'][roi_id]['signal']
        roi_c_trace = trace_dict['roi'][roi_id]['crosstalk']
        neuropil_s_trace = trace_dict['neuropil'][roi_id]['signal']
        neuropil_c_trace = trace_dict['neuropil'][roi_id]['crosstalk']

        is_valid = True

        for test_trace in (roi_c_trace,
                           neuropil_s_trace,
                           neuropil_c_trace):
            if test_trace.shape != roi_s_trace.shape:
                is_valid = False
                break

        if is_valid:
            for test_trace in (roi_s_trace,
                               roi_c_trace,
                               neuropil_s_trace,
                               neuropil_c_trace):

                if np.isnan(test_trace).any():
                    is_valid = False

        output_dict[roi_id] = is_valid

    return output_dict


def find_independent_events(signal_events, crosstalk_events, window=2):
    """
    Calculate independent events between signal_events and crosstalk_events.

    The algorithm uses window to extend the range of event matches, such that
    if an event happens at time t in the signal and time t+window in the
    crosstalk, they are *not* considered independent events. If window=0,
    then any events that are not exact matches (i.e. occurring at the same
    time point) will be considered independent events.

    Args:
        signal_events: a dict
            signal_events['trace'] is an array of the trace flux
                                   values of the signal channel

            signal_events['events'] is an array of the timestamp
                                    indices of the signal channel

        crosstalk_events: a dict (same structure as signal_events)

        window (int): the amount of blurring to use (default=2)

    Returns:
        independent_events: a dict of events that were in signal_events,
                            but not crosstalk_events +/- window

            independent_events['trace'] is an array of the trace flux values
            indpendent_events['events'] is an array of the timestamp indices
    """
    blurred_crosstalk = np.unique(np.concatenate([crosstalk_events['events']+ii
                                                  for ii in
                                                  np.arange(-window,
                                                            window+1)]))

    valid_signal_events = np.where(np.logical_not(
                                   np.isin(signal_events['events'],
                                           blurred_crosstalk)))
    return {'trace': signal_events['trace'][valid_signal_events],
            'events': signal_events['events'][valid_signal_events]}


def validate_cell_crosstalk(signal_events, crosstalk_events, window=2):
    """
    Determine if an ROI is a valid cell or a ghost based on the
    events detected in the signal and crosstalk channels

    Args:
        signal_events: a dict
            signal_events['trace'] is an array of the trace
                                   flux values of the signal channel

            signal_events['events'] is an array of the timestamp
                                    indices of the signal channel

        crosstalk_events: a dict (same structure as signal_events)

        window (int): the amount of blurring to use in
                      find_independent_events (default=2)

    Returns:
        is_valid_roi : a boolean that is true if there are
                       any independent events in the signal channel

        independent_events: a dict of events that were in signal_events,
                            but not crosstalk_events +/- window

            independent_events['trace'] is an array of the trace flux values
            indpendent_events['events'] is an array of the timestamp indices

    """

    independent_events = find_independent_events(signal_events,
                                                 crosstalk_events,
                                                 window=window)
    is_valid_roi = False
    if len(independent_events['events']) > 0:
        is_valid_roi = True
    return is_valid_roi, independent_events
