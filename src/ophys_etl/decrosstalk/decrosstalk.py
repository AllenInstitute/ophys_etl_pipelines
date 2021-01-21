import numpy as np

import ophys_etl.decrosstalk.decrosstalk_utils as d_utils
import ophys_etl.decrosstalk.ica_utils as ica_utils
import ophys_etl.decrosstalk.io_utils as io_utils
import ophys_etl.decrosstalk.active_traces as active_traces

import logging

logger = logging.getLogger(__name__)


def get_raw_traces(signal_plane, ct_plane):
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
    A dict of raw traces such that
        output['roi'][roi_id]['signal'] is the raw signal trace for ROI

        output['roi'][roi_id]['crosstalk'] is the raw crosstalk
                                           trace for ROI

        output['neuropil'][roi_od]['signal'] is the raw signal trace for
                                             the neuropil around ROI

        output['neuropil'][roi_id]['crosstalk'] is the raw crosstalk trace
                                                for the neuropil around ROI
    """

    signal_traces = signal_plane.movie.get_trace(signal_plane.roi_list)
    crosstalk_traces = ct_plane.movie.get_trace(signal_plane.roi_list)

    output = {}
    output['roi'] = {}
    output['neuropil'] = {}
    for roi_id in signal_traces['roi'].keys():
        _roi = {}
        _neuropil = {}

        _roi['signal'] = signal_traces['roi'][roi_id]
        _roi['crosstalk'] = crosstalk_traces['roi'][roi_id]
        output['roi'][roi_id] = _roi

        _neuropil['signal'] = signal_traces['neuropil'][roi_id]
        _neuropil['crosstalk'] = crosstalk_traces['neuropil'][roi_id]
        output['neuropil'][roi_id] = _neuropil

    return output


def unmix_ROI(roi_traces, seed=None, iters=10):
    """
    Unmix the signal and crosstalk traces for a single ROI

    Parameters
    ----------
    roi_traces is a dict that such that
        roi_traces['signal'] is a numpy array
                             containing the signal trace
        roi_traces['crosstalk'] is a numpy array
                                containing the crosstalk trace

    seed is an int used to seed the random number generator
    that sklearn.decompositions.FastICA uses

    iters is an int indicating the number of iterations of
    FastICA to run before giving up on convegence.

    Returns
    -------
    A dict such that
        output['mixing_matrix'] -- the mixing matrix that transforms
                                   the unmixed signals back into the data
        output['signal'] -- is the unmixed signal
        output['crosstalk'] -- is the unmixed crosstalk
        output['use_avg_mixing_matrix'] -- a boolean; if True, ICA
                                           did not actually converge;
                                           we must discard these results
                                           and unmix the signal and
                                           crosstalk using the average
                                           mixing matrix for the plane
    """

    ica_input = np.array([roi_traces['signal'], roi_traces['crosstalk']])

    (unmixed_signals,
     mixing_matrix,
     roi_demixed) = ica_utils.run_ica(ica_input,
                                      seed=seed,
                                      iters=iters)

    assert unmixed_signals.shape == ica_input.shape

    output = {}
    output['mixing_matrix'] = mixing_matrix
    output['signal'] = unmixed_signals[0, :]
    output['crosstalk'] = unmixed_signals[1, :]
    output['use_avg_mixing_matrix'] = not roi_demixed

    return output


def unmix_all_ROIs(raw_roi_traces, seed_lookup=None):
    """
    Unmix all of the ROIs in this DecrosstalkingOphysPlane.

    Parameters
    ----------
    raw_roi_traces is a dict
        raw_roi_traces['roi'][roi_id]['signal'] is the raw signal
                                                for the ROI

        raw_roi_traces['roi'][roi_id]['crosstalk'] is the raw crosstalk
                                                   for the ROI

        raw_roi_traces['neuropil'][roi_id]['signal'] is the raw signal
                                                   for the neuropil around
                                                   the ROI

        raw_roi_traces['neuropil'][roi_id]['crosstalk'] is the raw
                                                   crosstalk for the
                                                   neuropil around the ROI

    seed_lookup is a dict that maps roi_id to a seed for np.RandomState

    Returns
    -------
    A dict such that

        output['roi'][roi_id]['mixing_matrix'] -- the ROI's mixing matrix
        output['roi'][roi_id]['signal'] -- the ROI's unmixed signal
        output['roi'][roi_id]['crosstalk'] -- the ROI's unmixed crosstalk
        output['roi'][roi_id]['use_avg_mixing_matrix'] -- a boolean

                If True, the ROI was demixed using the average mixing
                matrix for the DecrosstalkingOphysPlane. In that case,
                the unconverged mixing_matrix, signal, and crosstalk
                will be stored in 'poorly_converged_mixing_matrix',
                'poorly_converged_signal' and 'poorly_converged_crosstalk'

        output['neuropil'][roi_id]['signal'] -- neuropil's unmixed signal
        output['neuropil'][roi_id]['crosstalk'] -- neuropil's unmixed
                                                   crosstalk
    """

    output = {}
    output['roi'] = {}
    output['neuropil'] = {}

    # first pass naively unmixing ROIs with ICA
    for roi_id in raw_roi_traces['roi'].keys():

        unmixed_roi = unmix_ROI(raw_roi_traces['roi'][roi_id],
                                seed=seed_lookup[roi_id],
                                iters=10)

        _out = {}
        if not unmixed_roi['use_avg_mixing_matrix']:
            _out = unmixed_roi
        else:
            _out = {'use_avg_mixing_matrix': True}
            for k in unmixed_roi.keys():
                if k == 'use_avg_mixing_matrix':
                    continue
                _out['poorly_converged_%s' % k] = unmixed_roi[k]
        output['roi'][roi_id] = _out

    # calculate avg mixing matrix from successful iterations
    _out = output['roi']
    alpha_arr = np.array([min(_out[roi_id]['mixing_matrix'][0, 0],
                              _out[roi_id]['mixing_matrix'][0, 1])
                          for roi_id in _out.keys()
                          if not _out[roi_id]['use_avg_mixing_matrix']])
    beta_arr = np.array([min(_out[roi_id]['mixing_matrix'][1, 0],
                             _out[roi_id]['mixing_matrix'][1, 1])
                         for roi_id in _out.keys()
                         if not _out[roi_id]['use_avg_mixing_matrix']])

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

        output['neuropil'][roi_id] = {}
        output['neuropil'][roi_id]['signal'] = unmixed_signals[0, :]
        output['neuropil'][roi_id]['crosstalk'] = unmixed_signals[1, :]

    return True, output


def get_trace_events(trace_dict,
                     trace_threshold_params={'len_ne': 20,
                                             'th_ag': 14}):
    """
    trace_dict is a dict such that
        trace_dict[roi_id]['signal'] is the signal channel
        trace_dict[roi_id]['crosstalk'] is the crosstalk channel

    trace_threshold_params -- a dict of kwargs that need to be
    passed to active_traces.get_trace_events
        default: {'len_ne': 20,
                  'th_ag': 14}

    Returns
    -------
    out_dict such that
        out_dict[roi_id]['signal']['trace'] is the trace of
                                            the signal channel events

        out_dict[roi_id]['signal']['events'] is the timestep events of
                                             the signal channel events

        out_dict[roi_id]['crosstalk']['trace'] is the trace of
                                               the signal channel events

        out_dict[roi_id]['crosstalk']['events'] is the timestep events of
                                                the signal channel events
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

    output = {}
    for i_roi, roi_id in enumerate(roi_id_list):
        local_dict = {}
        local_dict['signal'] = {}
        local_dict['crosstalk'] = {}
        local_dict['signal']['trace'] = sig_dict['trace'][i_roi]
        local_dict['signal']['events'] = sig_dict['events'][i_roi]
        local_dict['crosstalk']['trace'] = ct_dict['trace'][i_roi]
        local_dict['crosstalk']['events'] = ct_dict['events'][i_roi]
        output[roi_id] = local_dict

    return output


def get_crosstalk_data(trace_dict, events_dict):
    """
    trace_dict that contains
        trace_dict[roi_id]['signal']
        trace_dict[roi_id]['crosstalk']

    events_dict that contains
        events_dict[roi_id]['signal']['trace']
        events_dict[roi_id]['signal']['events']
        events_dict[roi_id]['crosstalk']['trace']
        events_dict[roi_id]['crosstalk']['events']

    returns a dict keyed on roi_id with 100*slope relating
    signal to crosstalk
    """
    output = {}
    for roi_id in trace_dict.keys():
        signal = events_dict[roi_id]['signal']['trace']
        full_crosstalk = trace_dict[roi_id]['crosstalk']
        crosstalk = full_crosstalk[events_dict[roi_id]['signal']['events']]
        results = d_utils.get_crosstalk_data(signal, crosstalk)
        output[roi_id] = 100*results['slope']
    return output


def run_decrosstalk(signal_plane, ct_plane,
                    cache_dir=None, clobber=False,
                    new_style_output=False):
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

        unmixed_traces -- a dict such that
        unmixed_traces['roi'][roi_id]['mixing_matrix'] -- the ROI's
                                                       mixing matrix
        unmixed_traces['roi'][roi_id]['signal'] -- the ROI's unmixed signal
        unmixed_traces['roi'][roi_id]['crosstalk'] -- the ROI's unmixed
                                                       crosstalk
        unmixed_traces['roi'][roi_id]['use_avg_mixing_matrix'] -- a boolean

                If True, the ROI was demixed using the average mixing
                matrix for the DecrosstalkingOphysPlane. In that case,
                the unconverged mixing_matrix, signal, and crosstalk
                will be stored in 'poorly_converged_mixing_matrix',
                'poorly_converged_signal' and 'poorly_converged_crosstalk'

        unmixed_traces['neuropil'][roi_id]['signal'] -- neuropil's
                                                    unmixed signal
        unmixed_traces['neuropil'][roi_id]['crosstalk'] -- neuropil's
                                                     unmixed crosstalk
    """

    # kwargs for output classes that write QC output
    output_kwargs = {'cache_dir': cache_dir,
                     'signal_plane': signal_plane,
                     'crosstalk_plane': ct_plane,
                     'clobber': clobber}

    roi_flags = {}

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

    ###############################
    # extract raw traces

    raw_traces = get_raw_traces(signal_plane, ct_plane)

    if cache_dir is not None:
        if new_style_output:
            writer_class = io_utils.RawH5Writer
        else:
            writer_class = io_utils.RawH5WriterOld
        writer = writer_class(data=raw_traces, **output_kwargs)
        writer.run()
        del writer
        del writer_class

    raw_trace_validation = d_utils.validate_traces(raw_traces)
    if cache_dir is not None:
        if new_style_output:
            writer_class = io_utils.ValidJsonWriter
        else:
            writer_class = io_utils.ValidJsonWriterOld
        writer = writer_class(data=raw_trace_validation, **output_kwargs)
        writer.run()
        del writer
        del writer_class

    # remove invalid raw traces
    invalid_raw_trace = []
    for roi_id in raw_trace_validation:
        if not raw_trace_validation[roi_id]:
            invalid_raw_trace.append(roi_id)
            raw_traces['roi'].pop(roi_id)
            raw_traces['neuropil'].pop(roi_id)
    roi_flags[raw_key] += invalid_raw_trace

    if len(raw_traces['roi']) == 0:
        msg = 'No raw traces were valid when applying '
        msg += 'decrosstalk to ophys_experiment_id: '
        msg += '%d (%d)' % (signal_plane.experiment_id,
                            ct_plane.experiment_id)
        logger.error(msg)
        return roi_flags, {}

    #########################################
    # detect activity in raw traces

    raw_trace_events = get_trace_events(raw_traces['roi'])

    # For each ROI, calculate a random seed based on the flux
    # in all timestamps *not* chosen as events (presumably,
    # random noise)
    roi_to_seed = {}
    two_to_32 = 2**32
    for roi_id in raw_trace_events:
        flux_mask = np.ones(len(raw_traces['roi'][roi_id]['signal']),
                            dtype=bool)
        flux_mask[raw_trace_events[roi_id]['signal']['events']] = False
        _flux = np.abs(raw_traces['roi'][roi_id]['signal'][flux_mask])
        flux_sum = np.round(_flux.sum()).astype(int)
        roi_to_seed[roi_id] = flux_sum % two_to_32

    if cache_dir is not None:
        if new_style_output:
            writer_class = io_utils.RawATH5Writer
        else:
            writer_class = io_utils.RawATH5WriterOld
        writer = writer_class(data=raw_trace_events, **output_kwargs)
        writer.run()
        del writer
        del writer_class

    raw_trace_crosstalk_ratio = get_crosstalk_data(raw_traces['roi'],
                                                   raw_trace_events)

    # remove ROIs with invalid active raw traces
    roi_id_list = list(raw_trace_events.keys())
    for roi_id in roi_id_list:
        signal = raw_trace_events[roi_id]['signal']['trace']
        if len(signal) == 0 or np.isnan(signal).any():
            roi_flags[raw_active_key].append(roi_id)
            raw_trace_events.pop(roi_id)
            raw_traces['roi'].pop(roi_id)
            raw_traces['neuropil'].pop(roi_id)

    del raw_trace_events

    ###########################################################
    # use Independent Component Analysis to separate out signal
    # and crosstalk

    (ica_converged,
     unmixed_traces) = unmix_all_ROIs(raw_traces,
                                      seed_lookup=roi_to_seed)
    if cache_dir is not None:
        if new_style_output:
            writer_class = io_utils.OutH5Writer
        else:
            writer_class = io_utils.OutH5WriterOld
        writer = writer_class(data=unmixed_traces, **output_kwargs)
        writer.run()
        del writer
        del writer_class

    if not ica_converged:
        for roi_id in unmixed_traces['roi']:
            roi_flags[unmixed_key].append(roi_id)

        msg = 'ICA did not converge for any ROIs when '
        msg += 'applying decrosstalk to ophys_experiment_id: '
        msg += '%d (%d)' % (signal_plane.experiment_id,
                            ct_plane.experiment_id)
        logger.error(msg)
        return roi_flags, unmixed_traces

    unmixed_trace_validation = d_utils.validate_traces(unmixed_traces)
    if cache_dir is not None:
        if new_style_output:
            writer_class = io_utils.OutValidJsonWriter
        else:
            writer_class = io_utils.OutValidJsonWriterOld
        writer = writer_class(data=unmixed_trace_validation,
                              **output_kwargs)
        writer.run()
        del writer
        del writer_class

    # remove invalid unmixed traces
    invalid_unmixed_trace = []
    for roi_id in unmixed_trace_validation:
        if not unmixed_trace_validation[roi_id]:
            invalid_unmixed_trace.append(roi_id)
            unmixed_traces['roi'].pop(roi_id)
            unmixed_traces['neuropil'].pop(roi_id)
    roi_flags[unmixed_key] += invalid_unmixed_trace

    if len(unmixed_traces['roi']) == 0:
        msg = 'No unmixed traces were valid when applying '
        msg += 'decrosstalk to ophys_experiment_id: '
        msg += '%d (%d)' % (signal_plane.experiment_id,
                            ct_plane.experiment_id)
        logger.error(msg)
        return roi_flags, unmixed_traces

    ###################################################
    # Detect activity in unmixed traces

    unmixed_trace_events = get_trace_events(unmixed_traces['roi'])

    # Sometimes, unmixed_trace_events will return an array of NaNs.
    # Until we can debug that behavior, we will log those errors,
    # store the relevaten ROIs as decrosstalk_invalid_unmixed_trace,
    # and cull those ROIs from the data

    invalid_active_trace = {}
    invalid_active_trace['signal'] = []
    invalid_active_trace['crosstalk'] = []
    for roi_id in unmixed_trace_events.keys():
        local_traces = unmixed_trace_events[roi_id]
        for channel in ('signal', 'crosstalk'):
            nan_trace = np.isnan(local_traces[channel]['trace']).any()
            nan_events = np.isnan(local_traces[channel]['events']).any()
            if nan_trace or nan_events:
                invalid_active_trace[channel].append(roi_id)

    if cache_dir is not None:
        if new_style_output:
            writer_class = io_utils.OutATH5Writer
        else:
            writer_class = io_utils.OutATH5WriterOld
        writer = writer_class(data=unmixed_trace_events,
                              **output_kwargs)
        writer.run()
        del writer
        del writer_class

    n_sig = len(invalid_active_trace['signal'])
    n_ct = len(invalid_active_trace['crosstalk'])
    if n_sig > 0 or n_ct > 0:
        msg = 'ophys_experiment_id: %d (%d) ' % (signal_plane.experiment_id,
                                                 ct_plane.experiment_id)
        msg += 'had ROIs with active event channels that contained NaNs'
        logger.error(msg)

        # remove ROIs with NaNs in their independent signal events
        # from the data being processed
        for roi_id in invalid_active_trace['signal']:
            roi_flags[unmixed_active_key].append(roi_id)
            unmixed_trace_events.pop(roi_id)
            unmixed_traces['roi'].pop(roi_id)
            unmixed_traces['neuropil'].pop(roi_id)

    if cache_dir is not None:
        writer = io_utils.InvalidATJsonWriter(data=invalid_active_trace,
                                              **output_kwargs)
        writer.run()
        del writer

    unmixed_ct_ratio = get_crosstalk_data(unmixed_traces['roi'],
                                          unmixed_trace_events)

    ########################################################
    # For each ROI, assess whether or not it is a "ghost"
    # (i.e. whether any of its activity is due to the signal,
    # independent of the crosstalk; if not, it is a ghost)

    independent_events = {}
    ghost_roi_id = []
    for roi_id in unmixed_trace_events:
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

    if cache_dir is not None:
        if new_style_output:
            writer_class = io_utils.ValidCTH5Writer
        else:
            writer_class = io_utils.ValidCTH5WriterOld
        writer = writer_class(data=independent_events,
                              **output_kwargs)
        writer.run()
        del writer
        del writer_class

        crosstalk_ratio = {}
        for roi_id in unmixed_ct_ratio:
            _out = {'raw': raw_trace_crosstalk_ratio[roi_id],
                    'unmixed': unmixed_ct_ratio[roi_id]}
            crosstalk_ratio[roi_id] = _out

        if new_style_output:
            writer_class = io_utils.CrosstalkJsonWriter
        else:
            writer_class = io_utils.CrosstalkJsonWriterOld
        writer = writer_class(data=crosstalk_ratio, **output_kwargs)
        writer.run()
        del writer
        del writer_class

    writer = io_utils.InvalidFlagWriter(data=roi_flags,
                                        **output_kwargs)
    writer.run()

    return roi_flags, unmixed_traces
