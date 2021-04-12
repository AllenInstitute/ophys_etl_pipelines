import h5py
import json
import os
import numpy as np
import pathlib
import ophys_etl.modules.decrosstalk.decrosstalk_types as dc_types


def _write_data_to_h5(file_handle, data_dict):
    """
    Utility to write nested dicts to h5py files

    This will result in an hdf5 file in which nested dicts
    appear as nested directories, i.e.

    data['a'] = True
    data['b'] = {'c': False, 'd':np.array([1,2,3,4])}

    would result in and hdf5 file with

    data['a'] = True
    data['b/c'] = False
    data['b/d'] = np.array([1,2,3,4])

    Parameters
    ----------
    file_handle -- a file handle pointing to the file in
    which we are writing

    data_dict -- the dict of data to be written
    """

    key_list = list(data_dict.keys())
    for key in key_list:
        value = data_dict[key]
        if hasattr(value, 'keys'):
            group = file_handle.create_group(str(key))
            _write_data_to_h5(group, value)
        else:

            # sometimes empty arrays of objects get through here
            if hasattr(value, '__len__'):
                if len(value) == 0:
                    value = np.array([], dtype=int)
            file_handle.create_dataset(str(key), data=value)
    return None


def write_to_h5(file_name, data_dict, clobber=False):
    """
    Write a nested dict to an HDF5 file, treating each level of
    key as a new directory in the HDF5 file's structure.

    e.g.
    data_dict['a'] = True
    data_dict['b'] = {'c': False, 'd':np.array([1,2,3,4])}

    would result in and hdf5 file with

    hdf5_data['a'] = True
    hdf5_data['b/c'] = False
    hdf5_data['b/d'] = np.array([1,2,3,4])


    Parameters
    -----------
    file_name -- the name of the HDF5 file to write

    data_dict -- the dict of data to write

    clobber -- boolean; if False, will not overwrite an existing file
    (default=False)
    """
    if not hasattr(data_dict, 'keys'):
        msg = '\nInput to write_to_h5 not a dict\n'
        msg += 'Input was %s' % str(type(data_dict))
        raise RuntimeError(msg)

    if os.path.exists(file_name) and not clobber:
        msg = "\n%s\nalready exists; " % file_name
        msg += "use clobber=True to overwrite"
        raise RuntimeError(msg)
    with h5py.File(file_name, mode='w') as file_handle:
        _write_data_to_h5(file_handle, data_dict)
    return None


def write_basic_json(out_fname, data, clobber=False):
    """
    Write data to a json file

    Parameters
    ----------
    out_fname -- name of the file to write

    data -- data to be written

    clobber -- boolean; if True, overwrite preexisting
    out_fname; otherwise, raise an exception if out_fname
    already exists (default=False)
    """
    if not clobber and os.path.exists(out_fname):
        raise RuntimeError("\n%s\nalready exists" % out_fname)
    with open(out_fname, 'w') as out_file:
        out_file.write(json.dumps(data, indent=2, sort_keys=True))


# The classes below are implemented to allow us to quickly
# switch between different output data models (the "old style"
# around which the prototype was written and the "new style"
# which will hopefully be easier to work with in the future).
#
# Each class writes a different output file. They all accept
# the same arguments in their constructors.
#
# cache_dir -- the parent output dir for the ophys_experimental_session
# signal_plane -- the OphysPlane of the "signal"
# crosstalk_plane -- the OphysPlane of the "crosstalk"
# clobber -- a boolean indicating whether or not to overwrite existing files
# data -- whatever data needs to be written in the output file.
#
# Each class implements its own run() method which actually writes
# the file(s) for which it is responsible.


class OutputWriter(object):
    def __init__(self,
                 cache_dir=None,
                 signal_plane=None,
                 crosstalk_plane=None,
                 data=None,
                 clobber=False):

        self.cache_dir = cache_dir
        self.signal_plane = signal_plane
        self.crosstalk_plane = crosstalk_plane
        self.data = data
        self.clobber = clobber

    def run(self):
        msg = "Calling OutputWriter.run(); should call child class"
        raise NotImplementedError(msg)


class OldOutputWriter(OutputWriter):

    @property
    def roi_dir(self):
        if not hasattr(self, '_roi_dir'):
            self._roi_dir = self.create_old_style_dir('roi')
        return self._roi_dir

    @property
    def neuropil_dir(self):
        if not hasattr(self, '_neuropil_dir'):
            self._neuropil_dir = self.create_old_style_dir('neuropil')
        return self._neuropil_dir

    def create_old_style_dir(self, prefix):
        e1 = self.signal_plane.experiment_id
        e2 = self.crosstalk_plane.experiment_id
        out_dir = os.path.join(self.cache_dir,
                               '%s_%d_%d' % (prefix,
                                             min(e1, e2),
                                             max(e1, e2)))

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if not os.path.isdir(out_dir):
            raise RuntimeError("\n%s\nis not a dir" % out_dir)

        return out_dir

    def get_sub_dir(self, prefix):
        if prefix == 'roi':
            out_dir = self.roi_dir
        elif prefix == 'neuropil':
            out_dir = self.neuropil_dir
        else:
            msg = "confused by prefix\n%s\n" % prefix
            msg += "in OldOutputWriter.get_sub_dir"
            raise RuntimeError(msg)
        return out_dir


class ValidJsonWriter(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_valid.json' %
                                 (self.signal_plane.experiment_id,
                                  self.crosstalk_plane.experiment_id))
        write_basic_json(out_fname, self.data, clobber=self.clobber)


class ValidJsonWriterOld(OldOutputWriter):
    def run(self):
        for dirname in (self.roi_dir, self.neuropil_dir):
            out_fname = os.path.join(dirname,
                                     '%d_valid.json' %
                                     self.signal_plane.experiment_id)
            write_basic_json(out_fname, self.data, clobber=self.clobber)


class RawH5Writer(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_raw.h5' %
                                 (self.signal_plane.experiment_id,
                                  self.crosstalk_plane.experiment_id))
        write_to_h5(out_fname, self.data, clobber=self.clobber)


class RawH5WriterOld(OldOutputWriter):

    def _write_to_h5(self, prefix):
        out_dir = self.get_sub_dir(prefix)
        roi_names = np.array([roi_id for roi_id in self.data[prefix].keys()])
        roi_names = np.sort(roi_names)

        local_data = np.zeros((2,
                               len(roi_names),
                               len(self.data[prefix][roi_names[0]]['signal'])),
                              dtype=float)

        for i_roi, roi_id in enumerate(roi_names):
            local_data[0, i_roi, :] = self.data[prefix][roi_id]['signal']
            local_data[1, i_roi, :] = self.data[prefix][roi_id]['crosstalk']

        results = {'roi_names': roi_names, 'data': local_data}
        out_fname = os.path.join(out_dir,
                                 '%d_raw.h5' %
                                 self.signal_plane.experiment_id)
        write_to_h5(out_fname,
                    results,
                    clobber=self.clobber)

    def run(self):
        for prefix in ('roi', 'neuropil'):
            self._write_to_h5(prefix)


class RawATH5Writer(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_raw_at.h5' %
                                 (self.signal_plane.experiment_id,
                                  self.crosstalk_plane.experiment_id))
        write_to_h5(out_fname, self.data, clobber=self.clobber)


class RawATH5WriterOld(OldOutputWriter):

    def _get_at_outname(self):
        out_fname = os.path.join(self.roi_dir,
                                 '%d_raw_at.h5' %
                                 self.signal_plane.experiment_id)
        return out_fname

    def run(self):
        output = {}
        for roi_id in self.data.keys():
            data = self.data[roi_id]
            output['%d_signal_trace' % roi_id] = data['signal']['trace']
            output['%d_signal_events' % roi_id] = data['signal']['events']
            signal_valid = True
            if np.isnan(data['signal']['trace']).any():
                signal_valid = False
            output['%d_signal_valid' % roi_id] = signal_valid
            output['%d_crosstalk_trace' % roi_id] = data['crosstalk']['trace']
            _events_key = '%d_crosstalk_events' % roi_id
            output[_events_key] = data['crosstalk']['events']

            if len(data['crosstalk']['trace']) == 0:
                f_trace = False
            else:
                f_trace = np.isfinite(data['crosstalk']['trace']).all()

            if len(data['crosstalk']['events']) == 0:
                f_ev = False
            else:
                f_ev = np.isfinite(data['crosstalk']['events']).all()

            f_ct = np.logical_and(f_trace, f_ev)
            output['%d_crosstalk_valid' % roi_id] = f_ct
        write_to_h5(self._get_at_outname(), output, clobber=self.clobber)


class OutATH5WriterOld(RawATH5WriterOld):

    def _get_at_outname(self):
        out_fname = os.path.join(self.roi_dir,
                                 '%d_out_at.h5' %
                                 self.signal_plane.experiment_id)
        return out_fname


class OutH5Writer(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_out.h5' %
                                 (self.signal_plane.experiment_id,
                                  self.crosstalk_plane.experiment_id))
        write_to_h5(out_fname, self.data, clobber=self.clobber)


class OutH5WriterOld(OldOutputWriter):
    def _get_channels(self, prefix):
        local_data = {}
        roi_names = np.array([roi_id for roi_id in self.data[prefix].keys()])
        roi_names = np.sort(roi_names)

        if len(roi_names) == 0:
            return {}

        if 'signal' not in self.data[prefix][roi_names[0]]:
            for roi in roi_names:
                if 'signal' in self.data[prefix][roi]:
                    raise RuntimeError("confused; some ROIs have signal; "
                                       "some do not")
            return {}
        n_t = len(self.data[prefix][roi_names[0]]['signal'])
        signal_data = np.zeros((len(roi_names), n_t), dtype=float)
        unclipped = np.zeros((len(roi_names), n_t), dtype=float)
        crosstalk_data = np.zeros((len(roi_names), n_t), dtype=float)
        demixed = np.zeros(len(roi_names), dtype=bool)
        for i_roi, roi_id in enumerate(roi_names):
            signal_data[i_roi, :] = self.data[prefix][roi_id]['signal']
            unclipped[i_roi, :] = self.data[prefix][roi_id]['unclipped_signal']
            crosstalk_data[i_roi, :] = self.data[prefix][roi_id]['crosstalk']
            _demixed = not self.data['roi'][roi_id]['use_avg_mixing_matrix']
            demixed[i_roi] = _demixed

        local_data['roi_names'] = roi_names
        local_data['data_signal'] = signal_data
        local_data['data_crosstalk'] = crosstalk_data
        local_data['roi_demixed'] = demixed
        local_data['unclipped_signal'] = unclipped

        for roi_id in roi_names:
            key_set = set(self.data[prefix][roi_id].keys())
            if 'poorly_converged_mixing_matrix' in key_set:
                for suffix in ('signal', 'crosstalk', 'mixing_matrix'):
                    field = 'poorly_converged_%s' % suffix
                    k = '%d/%s' % (roi_id, field)
                    v = self.data[prefix][roi_id][field]
                    local_data[k] = v

        return local_data

    def run(self):
        roi_data = self._get_channels('roi')

        if len(roi_data) == 0:
            return

        neuropil_data = self._get_channels('neuropil')

        local_mix = np.zeros((len(roi_data['roi_names']), 2, 2), dtype=float)
        for i_roi, roi_id in enumerate(roi_data['roi_names']):
            local_mix[i_roi, :, :] = self.data['roi'][roi_id]['mixing_matrix']

        out_dir = self.get_sub_dir('neuropil')

        out_fname = os.path.join(out_dir,
                                 '%d_out.h5' %
                                 self.signal_plane.experiment_id)

        neuropil_data['mixing_matrix'] = local_mix

        write_to_h5(out_fname,
                    neuropil_data,
                    clobber=self.clobber)

        out_dir = self.get_sub_dir('roi')
        out_fname = os.path.join(out_dir, '%d_out.h5' %
                                 self.signal_plane.experiment_id)

        roi_data['mixing_matrix'] = local_mix
        write_to_h5(out_fname, roi_data, clobber=self.clobber)


class OutATH5Writer(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_out_at.h5' %
                                 (self.signal_plane.experiment_id,
                                  self.crosstalk_plane.experiment_id))
        write_to_h5(out_fname, self.data, clobber=self.clobber)


class OutValidJsonWriter(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_out_valid.json' %
                                 (self.signal_plane.experiment_id,
                                  self.crosstalk_plane.experiment_id))
        write_basic_json(out_fname, self.data, clobber=self.clobber)


class OutValidJsonWriterOld(OldOutputWriter):
    def run(self):
        for prefix in ('roi', 'neuropil'):
            out_fname = os.path.join(self.get_sub_dir(prefix),
                                     '%d_out_valid.json' %
                                     (self.signal_plane.experiment_id))
            write_basic_json(out_fname, self.data, clobber=self.clobber)


class InvalidATJsonWriter(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_invalid_at.json' %
                                 (self.signal_plane.experiment_id,
                                  self.crosstalk_plane.experiment_id))
        write_basic_json(out_fname, self.data, clobber=self.clobber)


class CrosstalkJsonWriter(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_crosstalk.json' %
                                 (self.signal_plane.experiment_id,
                                  self.crosstalk_plane.experiment_id))
        write_basic_json(out_fname, self.data, clobber=self.clobber)


class CrosstalkJsonWriterOld(OldOutputWriter):
    def run(self):
        out_fname = os.path.join(self.get_sub_dir('roi'),
                                 '%d_crosstalk.json' %
                                 (self.signal_plane.experiment_id))

        local_data = {}
        for roi_id in self.data:
            local_data[roi_id] = [self.data[roi_id]['raw'],
                                  self.data[roi_id]['unmixed']]

        write_basic_json(out_fname, local_data, clobber=self.clobber)


class ValidCTH5Writer(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_valid_ct.h5' %
                                 (self.signal_plane.experiment_id,
                                  self.crosstalk_plane.experiment_id))

        write_to_h5(out_fname, self.data, clobber=self.clobber)


class ValidCTH5WriterOld(OldOutputWriter):
    def run(self):
        out_dir = self.get_sub_dir('roi')
        out_fname = os.path.join(out_dir,
                                 '%d_valid_ct.json' %
                                 self.signal_plane.experiment_id)
        local_data = {}
        for roi_id in self.data:
            local_data[roi_id] = self.data[roi_id]['is_a_cell']
        write_basic_json(out_fname, local_data, clobber=self.clobber)


class InvalidFlagWriter(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_invalid_flags.json' %
                                 (self.signal_plane.experiment_id,
                                  self.crosstalk_plane.experiment_id))
        write_basic_json(out_fname, self.data, clobber=self.clobber)


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
        The experiment_id of the plane paird with this plane

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

            if roi_id in invalid_raw:
                # If and ROI truly had an inactive raw trace,
                # it would never get to activity detection,
                # hence, if the roi_id is in invalid_raw_active,
                # it actually passed through raw trace extraction
                # and failed in activity detection
                if roi_id not in invalid_raw_active:
                    raw_valid = False
                    raw_active_valid = False

            if roi_id in invalid_unmixed_active:
                unmixed_active_valid = False

            if roi_id in invalid_unmixed:
                if roi_id not in invalid_unmixed_active:
                    unmixed_valid = False
                    unmixed_active_valid = False

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

                    is_ghost = False
                    if roi_id in ghost_set:
                        is_ghost = True

                    out_file.create_dataset(f'{parent_dir}/is_ghost',
                                            data=is_ghost)

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
                        out_file.create_dataset(set_name,
                                                data=data_dict['unclipped_signal'])  # noqa: E501

                    for sub_key in ('mixing_matrix',
                                    'poorly_converged_signal',
                                    'poorly_converged_crosstalk',
                                    'poorly_converged_mixing_matrix'):

                        if sub_key in data_dict:
                            set_name = f'{parent_dir}/{sub_key}'
                            out_file.create_dataset(set_name,
                                                    data=data_dict[sub_key])
    return None
