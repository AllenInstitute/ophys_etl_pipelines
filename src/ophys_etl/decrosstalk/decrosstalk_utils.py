import numpy as np

def find_independent_events(signal_events, crosstalk_events, window=2):
    ''' Calculate independent events between signal_events and crosstalk_events.

    The algorithm uses window to extend the range of event matches, such that
    if an event happens at time t in the signal and time t+window in the crosstalk,
    they are *not* considered independent events. If window=0, then any events
    that are not exact matches (i.e. occurring at the same time point) will be considered
    independent events.

    Args:
        signal_events: a dict
            signal_events['trace'] is an array of the trace flux values of the signal channel
            signal_events['events'] is an array of the timestamp indices of the signal channel
        crosstalk_events: a dict (same structure as signal_events)
        window (int): the amount of blurring to use (default=2)

    Returns:
        independent_events: a dict of events that were in signal_events, but not crosstalk_events +/- window
            independent_events['trace'] is an array of the trace flux values
            indpendent_events['events'] is an array of the timestamp indices
    '''
    blurred_crosstalk = np.unique(np.concatenate([crosstalk_events['events']+ii
                                                  for ii in np.arange(-window, window+1)]))

    valid_signal_events = np.where(np.logical_not(np.isin(signal_events['events'], blurred_crosstalk)))
    return {'trace': signal_events['trace'][valid_signal_events],
            'events': signal_events['events'][valid_signal_events]}


def validate_cell_crosstalk(signal_events, crosstalk_events, window=2):
    """
    Determine if an ROI is a valid cell or a ghost based on the events detected in the
    signal and crosstalk channels

    Args:
        signal_events: a dict
            signal_events['trace'] is an array of the trace flux values of the signal channel
            signal_events['events'] is an array of the timestamp indices of the signal channel
        crosstalk_events: a dict (same structure as signal_events)
        window (int): the amount of blurring to use in find_independent_events (default=2)

    Returns:
        is_valid_roi : a boolean that is true if there are any independent events in the signal channel

        independent_events: a dict of events that were in signal_events, but not crosstalk_events +/- window
            independent_events['trace'] is an array of the trace flux values
            indpendent_events['events'] is an array of the timestamp indices

    """

    independent_events = find_independent_events(signal_events, crosstalk_events, window=window)
    is_valid_roi = False
    if len(independent_events['events'])>0:
        is_valid_roi = True
    return is_valid_roi, independent_events
