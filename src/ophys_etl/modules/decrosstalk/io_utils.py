import h5py
import pathlib
import ophys_etl.modules.decrosstalk.decrosstalk_types as dc_types


def write_qc_data(file_name: pathlib.Path,
                  pair_id: int,
                  roi_flags: dict,
                  raw_traces: dc_types.ROISetDict,
                  invalid_raw_traces: dc_types.ROISetDict,
                  unmixed_traces: dc_types.ROISetDict,
                  invalid_unmixed_traces: dc_types.ROISetDict,
                  raw_events: dc_types.ROIEventSet,
                  invalid_raw_events: dc_types.ROIEventSet,
                  unmixed_events: dc_types.ROIEventSet,
                  invalid_unmixed_events: dc_types.ROIEventSet):
    """
    Write an HDF5 file containing the quality control data for a pair of
    planes

    Parameters
    ----------
    file_name: pathlib.Path
        The path to the HDF5 file that is being written

    pair_id: int
        The experiment_id of the plane paired with this plane

    roi_flags:
        A dict containing lists of all of the ROIs to which invalidity
        flags were applied

    raw_traces: decrosstalk_types.ROISetDict
        Raw traces associated with ROIs in this plane

    invalid_raw_traces: decrosstalk_types.ROISetDict
        Raw traces associated with ROIs in this plane
        that failed validity checks

    unmixed_traces: decrosstalk_types.ROISetDict
        Unmixed traces associated with ROIs in this plane

    invalid_unmixed_traces: decrosstalk_types.ROISetDict
        Unmixed traces associated with ROIs in this plane
        that failed validity checks

    raw_events: decrosstalk_types.ROIEventSet
        Active timestamps associated with raw traces

    invalid_raw_events: decrosstalk_types.ROIEventSet
        Active timestamps associated with raw traces
        that failed validity checks

    unmixed_events: decrosstalk_typs.ROIEventSet
        Active timestamps associated with unmixed traces

    invalid_unmixed_events: decrosstalk_typs.ROIEventSet
        Active timestamps associated with unmixed traces
        that failed validity checks


    Returns
    -------
    None
        Just writes HDF5 file
    """

    ghost_set = set(roi_flags['decrosstalk_ghost'])

    all_roi_id = set(raw_traces['roi'].keys())
    all_roi_id = all_roi_id.union(set(invalid_raw_traces['roi'].keys()))

    invalid_raw = set(invalid_raw_traces['roi'].keys())
    invalid_raw_active = set(invalid_raw_events.keys())
    invalid_unmixed = set(invalid_unmixed_traces['roi'].keys())
    invalid_unmixed_active = set(invalid_unmixed_events.keys())

    with h5py.File(file_name, 'w') as out_file:

        out_file.create_dataset('paired_plane', data=pair_id)

        # write flags indicating the validity of the datasets
        # associated with each ROI
        for roi_id in all_roi_id:
            raw_valid = True
            raw_active_valid = True
            unmixed_valid = True
            unmixed_active_valid = True

            if roi_id in invalid_raw_active:
                raw_active_valid = False
                unmixed_valid = False
                unmixed_active_valid = False

            if roi_id in invalid_raw:
                # If and ROI truly had an inactive raw trace,
                # it would never get to activity detection,
                # hence, if the roi_id is in invalid_raw_active,
                # it actually passed through raw trace extraction
                # and failed in activity detection
                if roi_id not in invalid_raw_active:
                    raw_valid = False
                    raw_active_valid = False
                    unmixed_valid = False
                    unmixed_active_valid = False

            if roi_id in invalid_unmixed_active:
                unmixed_active_valid = False

            if roi_id in invalid_unmixed:
                if roi_id not in invalid_unmixed_active:
                    unmixed_valid = False
                    unmixed_active_valid = False

            is_ghost = False
            if roi_id in ghost_set:
                is_ghost = True

            out_file.create_dataset(f'ROI/{roi_id}/is_ghost',
                                    data=is_ghost)

            out_file.create_dataset(f'ROI/{roi_id}/valid_raw_trace',
                                    data=raw_valid)

            out_file.create_dataset(f'ROI/{roi_id}/valid_raw_active_trace',
                                    data=raw_active_valid)

            out_file.create_dataset(f'ROI/{roi_id}/valid_unmixed_trace',
                                    data=unmixed_valid)

            out_file.create_dataset(f'ROI/{roi_id}/valid_unmixed_active_trace',
                                    data=unmixed_active_valid)

        # write the indices of events
        for parent_key, parent_data in zip(('raw',
                                            'unmixed',
                                            'raw',
                                            'unmixed'),
                                           (raw_events,
                                            unmixed_events,
                                            invalid_raw_events,
                                            invalid_unmixed_events)):

            for roi_id in parent_data.keys():
                if roi_id not in all_roi_id:
                    raise RuntimeError(f"{roi_id} not in all_roi_id")
                parent_dir = f'ROI/{roi_id}/roi/{parent_key}'
                data_dict = parent_data[roi_id]
                for channel_key in ('signal', 'crosstalk'):
                    channel_dict = data_dict[channel_key]
                    set_name = f'{parent_dir}/{channel_key}/events'
                    out_file.create_dataset(set_name,
                                            data=channel_dict['events'])

        # write raw traces
        for trace_set in (raw_traces, invalid_raw_traces):
            for data_key in ('roi', 'neuropil'):
                for roi_id in trace_set[data_key].keys():
                    if roi_id not in all_roi_id:
                        raise RuntimeError(f"{roi_id} not in all_roi_id")

                    parent_dir = f'ROI/{roi_id}/{data_key}/raw'

                    data_dict = trace_set[data_key][roi_id]
                    if 'signal' in data_dict:
                        set_name = f'{parent_dir}/signal/trace'
                        out_file.create_dataset(set_name,
                                                data=data_dict['signal'])
                    if 'crosstalk' in data_dict:
                        set_name = f'{parent_dir}/crosstalk/trace'
                        out_file.create_dataset(set_name,
                                                data=data_dict['crosstalk'])

        # write unmixed traces
        for trace_set in (unmixed_traces, invalid_unmixed_traces):
            for data_key in ('roi', 'neuropil'):
                for roi_id in trace_set[data_key].keys():

                    if roi_id not in all_roi_id:
                        raise RuntimeError(f"{roi_id} not in all_roi_id")

                    parent_dir = f'ROI/{roi_id}/{data_key}/unmixed'

                    data_dict = trace_set[data_key][roi_id]

                    if data_key == 'roi':
                        converged = (not data_dict['use_avg_mixing_matrix'])
                        out_file.create_dataset(f'{parent_dir}/converged',
                                                data=converged)

                    if 'signal' in data_dict:
                        set_name = f'{parent_dir}/signal/trace'
                        out_file.create_dataset(set_name,
                                                data=data_dict['signal'])
                    if 'crosstalk' in data_dict:
                        set_name = f'{parent_dir}/crosstalk/trace'
                        out_file.create_dataset(set_name,
                                                data=data_dict['crosstalk'])

                    if 'unclipped_signal' in data_dict:
                        set_name = f'{parent_dir}/unclipped_signal/trace'
                        tr = data_dict['unclipped_signal']
                        out_file.create_dataset(set_name,
                                                data=tr)

                    for sub_key in ('mixing_matrix',
                                    'poorly_converged_signal',
                                    'poorly_converged_crosstalk',
                                    'poorly_converged_mixing_matrix'):

                        if sub_key in data_dict:
                            set_name = f'{parent_dir}/{sub_key}'
                            out_file.create_dataset(set_name,
                                                    data=data_dict[sub_key])
    return None
