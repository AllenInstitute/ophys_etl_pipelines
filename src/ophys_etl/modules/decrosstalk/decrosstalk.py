import numpy as np

from typing import Tuple, Dict, List
import ophys_etl.modules.decrosstalk.decrosstalk_types as dc_types
from ophys_etl.modules.decrosstalk.ophys_plane import DecrosstalkingOphysPlane

import ophys_etl.modules.decrosstalk.decrosstalk_utils as d_utils
import ophys_etl.modules.decrosstalk.ica_utils as ica_utils
import ophys_etl.modules.decrosstalk.active_traces as active_traces

import logging

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def get_raw_traces(signal_plane: DecrosstalkingOphysPlane,
                   ct_plane: DecrosstalkingOphysPlane) -> dc_types.ROISetDict:
    """
    Get the raw signal and crosstalk traces comparing
    this plane to another plane

    Parameters
    ----------
    signal_plane -- an instance of DecrosstalkingOphysPlane which will
    be taken as the source of the signal

    ct_plane -- another instance of DecrosstalkingOphysPlane which will
    be taken as the source of the crosstalk

    Returns
    -------
    An decrosstalk_types.ROISetDict containing the raw trace data for
    the ROIs in the signal plane.
    """

    signal_traces = signal_plane.movie.get_trace(signal_plane.roi_list)
    crosstalk_traces = ct_plane.movie.get_trace(signal_plane.roi_list)

    output = dc_types.ROISetDict()

    for roi_id in signal_traces['roi'].keys():
        _roi = dc_types.ROIChannels()
        _neuropil = dc_types.ROIChannels()

        _roi['signal'] = signal_traces['roi'][roi_id]['signal']
        _roi['crosstalk'] = crosstalk_traces['roi'][roi_id]['signal']
        output['roi'][roi_id] = _roi

        _neuropil['signal'] = signal_traces['neuropil'][roi_id]['signal']
        _neuropil['crosstalk'] = crosstalk_traces['neuropil'][roi_id]['signal']
        output['neuropil'][roi_id] = _neuropil

    return output


def unmix_ROI(roi_traces: dc_types.ROIChannels,
              seed: int = None,
              iters: int = 10) -> dc_types.ROIChannels:
    """
    Unmix the signal and crosstalk traces for a single ROI

    Parameters
    ----------
    roi_traces -- a decrosstalking_types.ROIChannels containing
                  raw trace information for the ROI

    seed -- an int used to seed the random number generator
            that sklearn.decompositions.FastICA uses

    iters -- an int indicating the number of iterations of
             FastICA to run before giving up on convegence.

    Returns
    -------
    A decrosstalk_types.ROIChannels containing the unmixed trace data
    for the ROI
    """

    ica_input = np.array([roi_traces['signal'], roi_traces['crosstalk']])

    (unmixed_signals,
     mixing_matrix,
     roi_demixed) = ica_utils.run_ica(ica_input,
                                      iters,
                                      seed)

    assert unmixed_signals.shape == ica_input.shape

    output = dc_types.ROIChannels()
    output['mixing_matrix'] = mixing_matrix
    output['signal'] = unmixed_signals[0, :]
    output['crosstalk'] = unmixed_signals[1, :]
    output['use_avg_mixing_matrix'] = not roi_demixed

    return output


def unmix_all_ROIs(raw_roi_traces: dc_types.ROISetDict,
                   seed_lookup: Dict[int, int]
                   ) -> Tuple[bool, dc_types.ROISetDict]:
    """
    Unmix all of the ROIs in this DecrosstalkingOphysPlane.

    Parameters
    ----------
    raw_roi_traces -- a decrosstalk_types.ROISetDict containing the
                      raw trace data for all the ROIs to be unmixed

    seed_lookup -- a dict that maps roi_id to a seed for np.RandomState

    Returns
    -------
    A boolean indicating whether or not we were able to successfully
    unmix the ROIs in this input ROISetDict

    A decrosstalk_types.ROISetDict containing the unmixed trace data for the
    ROIs.

    Notes
    -----
    This method makes two attempts at decrosstalking each ROI using
    Independent Component Analysis. On the first attempt, each ROI
    (not neuropil) is decrosstalked as an independent entity. It is
    possible that this process will not converge for any given ROI.

    On the second attempt, an average demixing matrix is constructed
    from the demixing matrices of those ROIs for which the first attempt
    did converge. This average demixing matrix is to decrosstalk any
    ROIs for which the first attempt did not converge.

    Neuropils are then decrosstalked using the demixing matrix
    corresponding to their associated ROI (whether the ROI-specific
    demixing matrix or, in the case that the first attempt did not
    converge, the average demixing matrix).

    If the first attempt does not converge for any ROIs, there are no
    demixing matrices from which to construct an average demixing
    matrix and it is impossible to salvage any of the ROIs. In this case,
    the boolean that is the first returned object of this method will
    be set to False and the output ROISetDict will be emtpy.

    If decrosstalking converged for any ROIs (and an average demixing
    matrix is thus possible), that boolean will be set to True.

    The unmixed traces, however they were achieved, will be saved in
    the output ROISetDict.
    """

    output = dc_types.ROISetDict()

    # first pass naively unmixing ROIs with ICA
    for roi_id in raw_roi_traces['roi'].keys():

        unmixed_roi = unmix_ROI(raw_roi_traces['roi'][roi_id],
                                seed=seed_lookup[roi_id],
                                iters=10)

        if not unmixed_roi['use_avg_mixing_matrix']:
            _out = unmixed_roi
        else:
            _out = dc_types.ROIChannels()
            _out['use_avg_mixing_matrix'] = True
            for k in unmixed_roi.keys():
                if k == 'use_avg_mixing_matrix':
                    continue
                _out['poorly_converged_%s' % k] = unmixed_roi[k]
                _out[k] = np.NaN*np.zeros(unmixed_roi[k].shape, dtype=float)
        output['roi'][roi_id] = _out

    # calculate avg mixing matrix from successful iterations
    just_roi = output['roi']
    alpha_arr = np.array([min(just_roi[roi_id]['mixing_matrix'][0, 0],
                              just_roi[roi_id]['mixing_matrix'][0, 1])
                          for roi_id in just_roi.keys()
                          if not just_roi[roi_id]['use_avg_mixing_matrix']])
    beta_arr = np.array([min(just_roi[roi_id]['mixing_matrix'][1, 0],
                             just_roi[roi_id]['mixing_matrix'][1, 1])
                         for roi_id in just_roi.keys()
                         if not just_roi[roi_id]['use_avg_mixing_matrix']])

    assert alpha_arr.shape == beta_arr.shape
    if len(alpha_arr) == 0:
        return False, output

    mean_alpha = alpha_arr.mean()
    mean_beta = beta_arr.mean()
    mean_mixing_matrix = np.zeros((2, 2), dtype=float)
    mean_mixing_matrix[0, 0] = 1.0-mean_alpha
    mean_mixing_matrix[0, 1] = mean_alpha
    mean_mixing_matrix[1, 0] = mean_beta
    mean_mixing_matrix[1, 1] = 1.0-mean_beta
    inv_mean_mixing_matrix = np.linalg.inv(mean_mixing_matrix)

    for roi_id in raw_roi_traces['roi'].keys():
        inv_mixing_matrix = None
        mixing_matrix = None
        if not output['roi'][roi_id]['use_avg_mixing_matrix']:
            mixing_matrix = output['roi'][roi_id]['mixing_matrix']
            inv_mixing_matrix = np.linalg.inv(mixing_matrix)
        else:
            mixing_matrix = mean_mixing_matrix
            inv_mixing_matrix = inv_mean_mixing_matrix

            # assign missing outputs to ROIs that failed to converge
            output['roi'][roi_id]['mixing_matrix'] = mixing_matrix
            _roi_traces = raw_roi_traces['roi'][roi_id]
            unmixed_signals = np.dot(inv_mixing_matrix,
                                     np.array([_roi_traces['signal'],
                                               _roi_traces['crosstalk']]))
            output['roi'][roi_id]['signal'] = unmixed_signals[0, :]
            output['roi'][roi_id]['crosstalk'] = unmixed_signals[1, :]

        # assign outputs to 'neuropils'
        _np_traces = raw_roi_traces['neuropil'][roi_id]
        unmixed_signals = np.dot(inv_mixing_matrix,
                                 np.array([_np_traces['signal'],
                                           _np_traces['crosstalk']]))

        neuropil = dc_types.ROIChannels()
        neuropil['signal'] = unmixed_signals[0, :]
        neuropil['crosstalk'] = unmixed_signals[1, :]
        output['neuropil'][roi_id] = neuropil

    return True, output


def get_trace_events(trace_dict: dc_types.ROIDict,
                     trace_threshold_params: dict = {'len_ne': 20, 'th_ag': 14}
                     ) -> dc_types.ROIEventSet:
    """
    trace_dict -- a decrosstalk_types.ROIDict containing the trace data
                  for many ROIs to be analyzed

    trace_threshold_params -- a dict of kwargs that need to be
                              passed to active_traces.get_trace_events
                              default: {'len_ne': 20,
                                        'th_ag': 14}

    Returns
    -------
    A decrosstalk_type.ROIEventSet containing the active trace data
    for the ROIs
    """
    roi_id_list = list(trace_dict.keys())

    data_arr = np.array([trace_dict[roi_id]['signal']
                         for roi_id in roi_id_list])
    sig_dict = active_traces.get_trace_events(data_arr,
                                              trace_threshold_params)

    data_arr = np.array([trace_dict[roi_id]['crosstalk']
                         for roi_id in roi_id_list])
    ct_dict = active_traces.get_trace_events(data_arr,
                                             trace_threshold_params)

    output = dc_types.ROIEventSet()
    for i_roi, roi_id in enumerate(roi_id_list):
        local_channels = dc_types.ROIEventChannels()

        signal = dc_types.ROIEvents()
        signal['trace'] = sig_dict['trace'][i_roi]
        signal['events'] = sig_dict['events'][i_roi]
        local_channels['signal'] = signal

        crosstalk = dc_types.ROIEvents()
        crosstalk['trace'] = ct_dict['trace'][i_roi]
        crosstalk['events'] = ct_dict['events'][i_roi]
        local_channels['crosstalk'] = crosstalk

        output[roi_id] = local_channels

    return output


def _centered_rolling_mean(data: np.ndarray,
                           mask: np.ndarray,
                           window: int) -> Tuple[np.ndarray,
                                                 np.ndarray]:
    """
    Takes the rolling mean of an array of data with a specified width

    Parameters
    ----------
    data -- np.ndarray the array whose mean is to be taken

    mask -- a np.ndarray of booleans indicating which elements of data
            should be used to find the mean

    window -- int the size of the window (centered, if possible, on
              each element of data

    Returns
    -------
    An np.ndarray containing the rolling mean of data

    An np.ndarray containing the rolling stdev of the data
    """
    if len(data.shape) != 1:
        raise ValueError("_centered_rolling_mean is only meant "
                         "to run on 1-D np.ndarrays; you passed in "
                         "a %d-D array" % len(data.shape))
    half = window//2
    n_t = len(data)
    window_arr = np.ones(window, dtype=float)
    new_data = np.zeros(data.shape, dtype=float)
    new_data[mask] = data[mask]
    sum_ = np.convolve(window_arr, new_data)
    sum_sq = np.convolve(window_arr, new_data**2)
    mask_sum = np.convolve(window_arr, mask)

    i0 = window-1
    i1 = window-1+n_t-window

    # if mask has a run of `False`s that is longer than
    # `window`, the mean and std calculations below will
    # end up dividing by zero because there are zero valid
    # elements over which to take the mean and std. This
    # np.errstate context will suppress those warnings.
    with np.errstate(divide='ignore', invalid='ignore'):
        true_sum = np.zeros(n_t, dtype=float)
        true_sum[half:n_t-half] = sum_[i0:i1]
        true_sum[:half] = sum_[i0]
        true_sum[n_t-half:] = sum_[i1]

        true_sum_sq = np.zeros(n_t, dtype=float)
        true_sum_sq[half:n_t-half] = sum_sq[i0:i1]
        true_sum_sq[:half] = sum_sq[i0]
        true_sum_sq[n_t-half:] = sum_sq[i1]

        true_mask_sum = np.zeros(n_t, dtype=float)
        true_mask_sum[half:n_t-half] = mask_sum[i0:i1]
        true_mask_sum[:half] = mask_sum[i0]
        true_mask_sum[n_t-half:] = mask_sum[i1]

        mean = true_sum/true_mask_sum
        var = (true_sum_sq/true_mask_sum - mean*mean)
        var *= (true_mask_sum/(true_mask_sum-1))
        std = np.sqrt(var)

    return mean, std


def clean_negative_traces(trace_dict: dc_types.ROISetDict
                          ) -> dc_types.ROISetDict:
    """
    Parameters
    ----------
    trace_dict -- a decrosstalk_types.ROISetDict containing the traces
                   that need to be clipped

    Returns
    -------
    A decrosstalk_types.ROISetDict containing the clipped traces
    """
    active_trace_dict = get_trace_events(trace_dict['roi'])
    output_trace_dict = dc_types.ROISetDict()
    roi_id_list = trace_dict['roi'].keys()
    roi_id_list.sort()
    for roi_id in roi_id_list:
        n_t = len(trace_dict['roi'][roi_id]['signal'])

        # try to select only inactive timesteps;
        # if there are none, select all timesteps
        # (that would be an unexpected edge case)
        mask = np.ones(n_t, dtype=bool)
        mask[active_trace_dict[roi_id]['signal']['events']] = False
        if mask.sum() == 0:
            mask[:] = True

        for obj in ('roi', 'neuropil'):

            # because we are running this step before culling
            # traces that failed ICA, sometimes, ROIs without
            # valid neuropil traces will get through
            if roi_id not in trace_dict[obj]:
                continue

            # again: there may be traces with NaNs; these are
            # going to get failed, anyway; just ignore them
            # for now
            if np.isnan(trace_dict[obj][roi_id]['signal']).any():
                dummy = dc_types.ROIChannels()
                dummy['signal'] = trace_dict[obj][roi_id]['signal']
                output_trace_dict[obj][roi_id] = dummy
                continue

            (mean,
             std) = _centered_rolling_mean(trace_dict[obj][roi_id]['signal'],
                                           mask,
                                           1980)

            threshold = mean-2.0*std

            if (threshold < 0.0).any():
                msg = 'The unmixed "%s" trace for roi %d ' % (obj, roi_id)
                msg += 'contained negative flux values'
                logger.warning(msg)

            # if there were no valid timesteps when calculating the rolling
            # mean, set the threshold to a very negative value
            threshold = np.where(np.logical_not(np.isnan(threshold)),
                                 threshold,
                                 -999.0)

            if np.isnan(threshold).any():
                raise RuntimeError("There were NaNs in "
                                   "clean_negative_traces.threshold")

            # clip the trace at threshold
            trace = trace_dict[obj][roi_id]['signal']
            trace = np.where(trace > threshold, trace, threshold)
            channel = dc_types.ROIChannels()
            channel['signal'] = trace
            output_trace_dict[obj][roi_id] = channel

    return output_trace_dict


def run_decrosstalk(signal_plane: DecrosstalkingOphysPlane,
                    ct_plane: DecrosstalkingOphysPlane,
                    cache_dir: str = None, clobber: bool = False,
                    new_style_output: bool = False
                    ) -> Tuple[dict,
                               Tuple[dc_types.ROISetDict,
                                     dc_types.ROISetDict],
                               Tuple[dc_types.ROISetDict,
                                     dc_types.ROISetDict],
                               Tuple[dc_types.ROIEventSet,
                                     dc_types.ROIEventSet],
                               Tuple[dc_types.ROIEventSet,
                                     dc_types.ROIEventSet]
                               ]:
    """
    Actually run the decrosstalking pipeline, comparing two
    DecrosstalkingOphysPlanes

    Parameters
    ----------
    signal_plane -- the DecrosstalkingOphysPlane characterizing the
                    signal plane

    ct_plane -- the DecrosstalkingOphysPlane characterizing the crosstalk plane

    cache_dir -- the directory in which to write the QC output
    (if None, the output does not get written)

    clobber -- a boolean indicating whether or not to overwrite
    pre-existing output files (default: False)

    new_style_output -- a boolean (default: False)

    Returns
    -------
    roi_flags -- a dict listing the ROI IDs of ROIs that were
    ruled invalid for different reasons, namely:

        'decrosstalk_ghost' -- ROIs that are ghosts

        'decrosstalk_invalid_raw' -- ROIs with invalid
                                     raw traces

        'decrosstalk_invalid_raw_active' -- ROIs with invalid
                                            raw active traces

        'decrosstalk_invalid_unmixed' -- ROIs with invalid
                                         unmixed traces

        'decrosstalk_invalid_unmixed_active' -- ROIs with invalid
                                                unmixed active traces

    (raw_traces,
     invalid_raw_traces) -- two decrosstalk_types.ROISetDicts containing
                            the raw trace data for the ROIs and the
                            invalid raw traces

    (unmixed_traces,
     invalid_unmixed_traces) -- two decrosstalk_types.ROISetDicts containing
                                the unmixed trace data for the ROIs and then
                                invalid unmixed traces

    (raw_trace_events,
     invalid_raw_trace_events) -- two decrosstalk_types.ROIEventSets
                                  characterizing the active timestamps
                                  from the raw traces and the invalid
                                  active timestamps

    (unmixed_trace_events,
     invalid_unmixed_trace_events) -- two decrosstalk_types.ROIEventSets
                                      characterizing the active timestamps
                                      from the unmixed traces and the invalid
                                      unmixed trace events
    """
    raw_traces = dc_types.ROISetDict()
    unmixed_traces = dc_types.ROISetDict()
    raw_trace_events = dc_types.ROIEventSet()
    unmixed_trace_events = dc_types.ROIEventSet()

    invalid_raw_traces = dc_types.ROISetDict()
    invalid_unmixed_traces = dc_types.ROISetDict()
    invalid_raw_trace_events = dc_types.ROIEventSet()
    invalid_unmixed_trace_events = dc_types.ROIEventSet()

    roi_flags: Dict[str, List[int]] = {}

    ghost_key = 'decrosstalk_ghost'
    raw_key = 'decrosstalk_invalid_raw'
    raw_active_key = 'decrosstalk_invalid_raw_active'
    unmixed_key = 'decrosstalk_invalid_unmixed'
    unmixed_active_key = 'decrosstalk_invalid_unmixed_active'

    roi_flags[ghost_key] = []
    roi_flags[raw_key] = []
    roi_flags[unmixed_key] = []
    roi_flags[raw_active_key] = []
    roi_flags[unmixed_active_key] = []

    # If there are no ROIs in the signal plane,
    # just return a set of empty outputs
    if len(signal_plane.roi_list) == 0:
        return (roi_flags,
                (raw_traces, invalid_raw_traces),
                (unmixed_traces, invalid_unmixed_traces),
                (raw_trace_events, invalid_raw_trace_events),
                (unmixed_trace_events, invalid_unmixed_trace_events))

    ###############################
    # extract raw traces

    raw_traces = get_raw_traces(signal_plane, ct_plane)
    raw_trace_validation = d_utils.validate_traces(raw_traces)

    # remove invalid raw traces
    invalid_raw_trace_roi_id = []
    for roi_id in raw_trace_validation:
        if not raw_trace_validation[roi_id]:
            invalid_raw_trace_roi_id.append(roi_id)

            _roi = raw_traces['roi'].pop(roi_id)
            _neuropil = raw_traces['neuropil'].pop(roi_id)

            invalid_raw_traces['roi'][roi_id] = _roi
            invalid_raw_traces['neuropil'][roi_id] = _neuropil

    roi_flags[raw_key] += invalid_raw_trace_roi_id

    if len(raw_traces['roi']) == 0:
        msg = 'No raw traces were valid when applying '
        msg += 'decrosstalk to ophys_experiment_id: '
        msg += '%d (%d)' % (signal_plane.experiment_id,
                            ct_plane.experiment_id)
        logger.error(msg)

        return (roi_flags,
                (raw_traces, invalid_raw_traces),
                (unmixed_traces, invalid_unmixed_traces),
                (raw_trace_events, invalid_raw_trace_events),
                (unmixed_trace_events, invalid_unmixed_trace_events))

    #########################################
    # detect activity in raw traces

    raw_trace_events = get_trace_events(raw_traces['roi'])

    # For each ROI, calculate a random seed based on the flux
    # in all timestamps *not* chosen as events (presumably,
    # random noise)
    roi_to_seed = {}
    two_to_32 = 2**32
    for roi_id in raw_trace_events.keys():
        flux_mask = np.ones(len(raw_traces['roi'][roi_id]['signal']),
                            dtype=bool)
        if len(raw_trace_events[roi_id]['signal']['events']) > 0:
            flux_mask[raw_trace_events[roi_id]['signal']['events']] = False
        _flux = np.abs(raw_traces['roi'][roi_id]['signal'][flux_mask])
        flux_sum = np.round(_flux.sum()).astype(int)
        roi_to_seed[roi_id] = flux_sum % two_to_32

    # remove ROIs with invalid active raw traces
    roi_id_list = list(raw_trace_events.keys())
    for roi_id in roi_id_list:
        signal = raw_trace_events[roi_id]['signal']['trace']
        if len(signal) == 0 or np.isnan(signal).any():
            roi_flags[raw_active_key].append(roi_id)

            _events = raw_trace_events.pop(roi_id)
            _roi = raw_traces['roi'].pop(roi_id)
            _neuropil = raw_traces['neuropil'].pop(roi_id)

            invalid_raw_traces['roi'][roi_id] = _roi
            invalid_raw_traces['neuropil'][roi_id] = _neuropil
            invalid_raw_trace_events[roi_id] = _events

    # if there was no activity in the raw traces, return an
    # empty ROISetDict because none of the ROIs were valid
    if len(raw_traces['roi']) == 0:
        return (roi_flags,
                (raw_traces, invalid_raw_traces),
                (unmixed_traces, invalid_unmixed_traces),
                (raw_trace_events, invalid_raw_trace_events),
                (unmixed_trace_events, invalid_unmixed_trace_events))

    ###########################################################
    # use Independent Component Analysis to separate out signal
    # and crosstalk

    (ica_converged,
     unmixed_traces) = unmix_all_ROIs(raw_traces,
                                      roi_to_seed)

    # clip dips in signal channel
    clipped_traces = clean_negative_traces(unmixed_traces)

    # save old signal to 'unclipped_signal'
    # save new signal to 'signal'
    for obj in ('roi', 'neuropil'):
        for roi_id in unmixed_traces[obj].keys():
            s = unmixed_traces[obj][roi_id]['signal']
            unmixed_traces[obj][roi_id]['unclipped_signal'] = s
            s = clipped_traces[obj][roi_id]['signal']
            unmixed_traces[obj][roi_id]['signal'] = s

    if not ica_converged:
        for roi_id in unmixed_traces['roi'].keys():
            roi_flags[unmixed_key].append(roi_id)

        msg = 'ICA did not converge for any ROIs when '
        msg += 'applying decrosstalk to ophys_experiment_id: '
        msg += '%d (%d)' % (signal_plane.experiment_id,
                            ct_plane.experiment_id)
        logger.error(msg)

        return (roi_flags,
                (raw_traces, invalid_raw_traces),
                (unmixed_traces, invalid_unmixed_traces),
                (raw_trace_events, invalid_raw_trace_events),
                (unmixed_trace_events, invalid_unmixed_trace_events))

    unmixed_trace_validation = d_utils.validate_traces(unmixed_traces)

    # remove invalid unmixed traces
    invalid_unmixed_trace_roi_id = []
    for roi_id in unmixed_trace_validation:
        if not unmixed_trace_validation[roi_id]:
            invalid_unmixed_trace_roi_id.append(roi_id)

            _roi = unmixed_traces['roi'].pop(roi_id)
            _neuropil = unmixed_traces['neuropil'].pop(roi_id)

            invalid_unmixed_traces['roi'][roi_id] = _roi
            invalid_unmixed_traces['neuropil'][roi_id] = _neuropil

    roi_flags[unmixed_key] += invalid_unmixed_trace_roi_id

    if len(unmixed_traces['roi']) == 0:
        msg = 'No unmixed traces were valid when applying '
        msg += 'decrosstalk to ophys_experiment_id: '
        msg += '%d (%d)' % (signal_plane.experiment_id,
                            ct_plane.experiment_id)
        logger.error(msg)

        return (roi_flags,
                (raw_traces, invalid_raw_traces),
                (unmixed_traces, invalid_unmixed_traces),
                (raw_trace_events, invalid_raw_trace_events),
                (unmixed_trace_events, invalid_unmixed_trace_events))

    ###################################################
    # Detect activity in unmixed traces

    unmixed_trace_events = get_trace_events(unmixed_traces['roi'])

    # Sometimes, unmixed_trace_events will return an array of NaNs.
    # Until we can debug that behavior, we will log those errors,
    # store the relevaten ROIs as decrosstalk_invalid_unmixed_trace,
    # and cull those ROIs from the data

    invalid_active_trace: Dict[str, List[int]] = {}
    invalid_active_trace['signal'] = []
    invalid_active_trace['crosstalk'] = []
    active_trace_had_NaNs = False
    for roi_id in unmixed_trace_events.keys():
        local_traces = unmixed_trace_events[roi_id]
        for channel in ('signal', 'crosstalk'):
            is_valid = True
            if len(local_traces[channel]['trace']) == 0:
                is_valid = False
            else:
                nan_trace = np.isnan(local_traces[channel]['trace']).any()
                nan_events = np.isnan(local_traces[channel]['events']).any()
                if nan_trace or nan_events:
                    is_valid = False
                    active_trace_had_NaNs = True

            if not is_valid:
                invalid_active_trace[channel].append(roi_id)

    if active_trace_had_NaNs:
        msg = 'ophys_experiment_id: %d (%d) ' % (signal_plane.experiment_id,
                                                 ct_plane.experiment_id)
        msg += 'had ROIs with active event channels that contained NaNs'
        logger.error(msg)

    # remove ROIs with empty or NaN active trace signal channels
    # from the data being processed
    for roi_id in invalid_active_trace['signal']:
        roi_flags[unmixed_active_key].append(roi_id)

        _events = unmixed_trace_events.pop(roi_id)
        _roi = unmixed_traces['roi'].pop(roi_id)
        _neuropil = unmixed_traces['neuropil'].pop(roi_id)

        invalid_unmixed_traces['roi'][roi_id] = _roi
        invalid_unmixed_traces['neuropil'][roi_id] = _neuropil
        invalid_unmixed_trace_events[roi_id] = _events

    ########################################################
    # For each ROI, assess whether or not it is a "ghost"
    # (i.e. whether any of its activity is due to the signal,
    # independent of the crosstalk; if not, it is a ghost)

    independent_events = {}
    ghost_roi_id = []
    for roi_id in unmixed_trace_events.keys():
        signal = unmixed_trace_events[roi_id]['signal']
        crosstalk = unmixed_trace_events[roi_id]['crosstalk']

        (is_a_cell,
         ind_events) = d_utils.validate_cell_crosstalk(signal, crosstalk)

        local = {'is_a_cell': is_a_cell,
                 'independent_events': ind_events}

        independent_events[roi_id] = local
        if not is_a_cell:
            ghost_roi_id.append(roi_id)
    roi_flags[ghost_key] += ghost_roi_id

    return (roi_flags,
            (raw_traces, invalid_raw_traces),
            (unmixed_traces, invalid_unmixed_traces),
            (raw_trace_events, invalid_raw_trace_events),
            (unmixed_trace_events, invalid_unmixed_trace_events))
