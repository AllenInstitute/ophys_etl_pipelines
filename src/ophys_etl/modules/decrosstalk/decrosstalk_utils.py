from typing import Dict, Tuple
import numpy as np

import ophys_etl.modules.decrosstalk.decrosstalk_types as dc_types


def validate_traces(trace_dict: dc_types.ROISetDict) -> Dict[int, bool]:
    """
    Check a traces_dict for validity.
    Validity is defined as neuropil and roi traces having the same shape.
    No NaNs appearing in any trace.

    Parameters
    -----------
    trace_dict -- a decrosstalk_types.ROISetDict containing the active
                  traces to be validated

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


def find_independent_events(signal_events: dc_types.ROIEvents,
                            crosstalk_events: dc_types.ROIEvents,
                            window: int = 2) -> dc_types.ROIEvents:
    """
    Calculate independent events between signal_events and crosstalk_events.

    The algorithm uses window to extend the range of event matches, such that
    if an event happens at time t in the signal and time t+window in the
    crosstalk, they are *not* considered independent events. If window=0,
    then any events that are not exact matches (i.e. occurring at the same
    time point) will be considered independent events.

    Parameters
    ----------
    signal_events -- a decrosstalk_types.ROIEvents containing the active
                     traces from the signal channel

    crosstalk_events -- a decrosstalk_types.ROIEvents containing the active
                        traces from the crosstalk channel

    window -- an int specifying the amount of blurring to use (default=2)

    Returns
    -------
    independent_events -- a decrosstalking_types.ROIEvents containing
                          the active traces and events that were in
                          signal_events, but not crosstalk_events +/- window
    """
    blurred_crosstalk = np.unique(np.concatenate([crosstalk_events['events']+ii
                                                  for ii in
                                                  np.arange(-window,
                                                            window+1)]))

    valid_signal_events = np.where(np.logical_not(
                                   np.isin(signal_events['events'],
                                           blurred_crosstalk)))

    output = dc_types.ROIEvents()
    output['trace'] = signal_events['trace'][valid_signal_events]
    output['events'] = signal_events['events'][valid_signal_events]

    return output


def validate_cell_crosstalk(signal_events: dc_types.ROIEvents,
                            crosstalk_events: dc_types.ROIEvents,
                            window: int = 2) -> Tuple[bool,
                                                      dc_types.ROIEvents]:
    """
    Determine if an ROI is a valid cell or a ghost based on the
    events detected in the signal and crosstalk channels

    Parameters
    ----------
    signal_events -- a decrosstalk_types.ROIEvents containing the active
                     traces from the signal channel

    crosstalk_events -- a decrosstalk_types.ROIEvents containing the active
                        traces from the crosstalk channel

    window -- an int specifying the amount of blurring to use (default=2)

    Returns
    -------
    is_valid_roi -- a boolean indicating whether or not there were independent
                    events in signal_events

    independent_events -- a decrosstalking_types.ROIEvents containing
                          the active traces and events that were in
                          signal_events, but not crosstalk_events +/- window
    """

    independent_events = find_independent_events(signal_events,
                                                 crosstalk_events,
                                                 window=window)
    is_valid_roi = False
    if len(independent_events['events']) > 0:
        is_valid_roi = True
    return is_valid_roi, independent_events
