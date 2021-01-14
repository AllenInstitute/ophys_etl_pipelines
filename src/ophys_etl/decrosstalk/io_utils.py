import h5py
import json
import os
import numpy as np


def _write_data_to_h5(file_handle, data_dict):
    # utility to write nested dicts to h5py files
    # (this may already exist in allensdk; haven't checked)

    # this will create an hdf5 file in which nested dicts
    # appear as nested directories, i.e.
    #
    # data['a'] = True
    # data['b'] = {'c': False, 'd':np.array([1,2,3,4])}
    #
    # would result in and hdf5 file with
    #
    # data['a'] = True
    # data['b/c'] = False
    # data['b/d'] = np.array([1,2,3,4])

    key_list = list(data_dict.keys())
    for key in key_list:
        value = data_dict[key]
        if isinstance(value, dict):
            group = file_handle.create_group(str(key))
            _write_data_to_h5(group, value)
        else:
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
    if not isinstance(data_dict, dict):
        msg = '\nInput to write_to_h5 not a dict\n'
        msg += 'Input was %s' % str(type(data_dict))
        raise RuntimeError(msg)

    if os.path.exists(file_name) and not clobber:
        raise RuntimeError("\n%s\nalready exists; use clobber=True to overwrite" %
                           file_name)
    with h5py.File(file_name, mode='w') as file_handle:
        _write_data_to_h5(file_handle, data_dict)
    return None


def write_basic_json(out_fname, data, clobber=False):
    if not clobber and os.path.exists(out_fname):
        raise RuntimeError("\n%s\nalready exists" % out_fname)
    with open(out_fname, 'w') as out_file:
        out_file.write(json.dumps(data, indent=2, sort_keys=True))


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
        raise NotImplementedError("Calling OutputWriter.run(); should call child class")


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
                                             min(e1,e2),
                                             max(e1,e2)))

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
            raise RuntimeError("confused by prefix\n%s\nin OldOutputWriter.get_sub_dir" % prefix)
        return out_dir


class ValidJsonWriter(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_valid.json' % (self.signal_plane.experiment_id,
                                                       self.crosstalk_plane.experiment_id))
        write_basic_json(out_fname, self.data, clobber=self.clobber)


class ValidJsonWriterOld(OldOutputWriter):
    def run(self):
        for dirname in (self.roi_dir, self.neuropil_dir):
            out_fname = os.path.join(dirname, '%d_valid.json' % self.signal_plane.experiment_id)
            write_basic_json(out_fname, self.data, clobber=self.clobber)


class RawH5Writer(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_raw.h5' % (self.signal_plane.experiment_id,
                                                   self.crosstalk_plane.experiment_id))
        write_to_h5(out_fname, self.data, clobber=self.clobber)

class RawH5WriterOld(OldOutputWriter):

    def _write_to_h5(self, prefix):
        out_dir = self.get_sub_dir(prefix)
        roi_names = np.array([roi_id for roi_id in self.data[prefix].keys()])
        roi_names = np.sort(roi_names)
        local_data = np.zeros((2, len(roi_names), len(self.data[prefix][roi_names[0]]['signal'])), dtype=float)
        for i_roi, roi_id in enumerate(roi_names):
            local_data[0,i_roi,:] = self.data[prefix][roi_id]['signal']
            local_data[1,i_roi,:] = self.data[prefix][roi_id]['crosstalk']

        results = {'roi_names':roi_names, 'data':local_data}
        out_fname = os.path.join(out_dir, '%d_raw.h5' % self.signal_plane.experiment_id)
        write_to_h5(out_fname, {'roi_names':roi_names, 'data':local_data}, clobber=self.clobber)


    def run(self):
        for prefix in ('roi', 'neuropil'):
            self._write_to_h5(prefix)


class RawATH5Writer(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_raw_at.h5' % (self.signal_plane.experiment_id,
                                                    self.crosstalk_plane.experiment_id))
        write_to_h5(out_fname, self.data, clobber=self.clobber)


class RawATH5WriterOld(OldOutputWriter):

    def _get_at_outname(self):
        out_fname = os.path.join(self.roi_dir, '%d_raw_at.h5' % self.signal_plane.experiment_id)
        return out_fname

    def run(self):
        local_data = {}
        for roi_id in self.data.keys():
            local_data['%d_signal_trace' % roi_id] = self.data[roi_id]['signal']['trace']
            local_data['%d_signal_events' % roi_id] = self.data[roi_id]['signal']['events']
            local_data['%d_signal_valid' % roi_id] = True
            local_data['%d_crosstalk_trace' % roi_id] = self.data[roi_id]['crosstalk']['trace']
            local_data['%d_crosstalk_events' % roi_id] = self.data[roi_id]['crosstalk']['trace']
            local_data['%d_crosstalk_valid' % roi_id] = np.logical_and(np.isfinite(self.data[roi_id]['crosstalk']['trace']).all(),
                                                                       np.isfinite(self.data[roi_id]['crosstalk']['events']).all())
        write_to_h5(self._get_at_outname(), local_data, clobber=self.clobber)


class OutATH5WriterOld(RawATH5WriterOld):

    def _get_at_outname(self):
        out_fname = os.path.join(self.roi_dir, '%d_out_at.h5' % self.signal_plane.experiment_id)
        return out_fname


class OutH5Writer(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_out.h5' % (self.signal_plane.experiment_id,
                                                   self.crosstalk_plane.experiment_id))
        write_to_h5(out_fname, self.data, clobber=self.clobber)


class OutH5WriterOld(OldOutputWriter):
    def _get_channels(self, prefix):
        local_data = {}
        roi_names = np.array([roi_id for roi_id in self.data[prefix].keys()])
        roi_names = np.sort(roi_names)
        if len(roi_names) == 0:
           return {}
        is_valid = False
        if 'signal' not in self.data[prefix][roi_names[0]]:
            for roi in roi_names:
                if 'signal' in self.data[prefix][roi]:
                    raise RuntimeError("confused; some ROIs have signal; some do not")
            return {}
        n_t = len(self.data[prefix][roi_names[0]]['signal'])
        signal_data = np.zeros((len(roi_names), n_t), dtype=float)
        crosstalk_data = np.zeros((len(roi_names), n_t), dtype=float)
        demixed = np.zeros(len(roi_names), dtype=bool)
        for i_roi, roi_id in enumerate(roi_names):
            signal_data[i_roi,:] = self.data[prefix][roi_id]['signal']
            crosstalk_data[i_roi,:] = self.data[prefix][roi_id]['crosstalk']
            demixed = not self.data['roi'][roi_id]['use_avg_mixing_matrix']

        local_data['roi_names'] = roi_names
        local_data['data_signal'] = signal_data
        local_data['data_crosstalk'] = crosstalk_data
        local_data['roi_demixed'] = demixed
        return local_data

    def run(self):
        roi_data = self._get_channels('roi')

        if len(roi_data) == 0:
            return

        out_dir = self.get_sub_dir('neuropil')
        out_fname = os.path.join(out_dir, '%d_out.h5' % self.signal_plane.experiment_id)
        write_to_h5(out_fname, self._get_channels('neuropil'), clobber=self.clobber)

        out_dir = self.get_sub_dir('roi')
        out_fname = os.path.join(out_dir, '%d_out.h5' % self.signal_plane.experiment_id)
        local_mix = np.zeros((len(roi_data['roi_names']),2,2),dtype=float)
        for i_roi, roi_id in enumerate(roi_data['roi_names']):
            local_mix[i_roi,:,:] = self.data['roi'][roi_id]['mixing_matrix']
        roi_data['mixing_matrix'] = local_mix
        write_to_h5(out_fname, roi_data, clobber=self.clobber)


class OutATH5Writer(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_out_at.h5' % (self.signal_plane.experiment_id,
                                                   self.crosstalk_plane.experiment_id))
        write_to_h5(out_fname, self.data, clobber=self.clobber)


class OutValidJsonWriter(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_out_valid.json' % (self.signal_plane.experiment_id,
                                                           self.crosstalk_plane.experiment_id))
        write_basic_json(out_fname, self.data, clobber=self.clobber)


class OutValidJsonWriterOld(OldOutputWriter):
    def run(self):
        for prefix in ('roi', 'neuropil'):
            out_fname = os.path.join(self.get_sub_dir(prefix),
                                     '%d_out_valid.json' % (self.signal_plane.experiment_id))
            write_basic_json(out_fname, self.data, clobber=self.clobber)


class InvalidATJsonWriter(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                '%d_%d_invalid_at.json' % (self.signal_plane.experiment_id,
                                                           self.crosstalk_plane.experiment_id))
        write_basic_json(out_fname, self.data, clobber=self.clobber)


class CrosstalkJsonWriter(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                '%d_%d_crosstalk.json' % (self.signal_plane.experiment_id,
                                                          self.crosstalk_plane.experiment_id))
        write_basic_json(out_fname, self.data, clobber=self.clobber)


class CrosstalkJsonWriterOld(OldOutputWriter):
    def run(self):
        out_fname = os.path.join(self.get_sub_dir('roi'),
                                '%d_crosstalk.json' % (self.signal_plane.experiment_id))

        local_data = {}
        for roi_id in self.data:
            local_data[roi_id] = [self.data[roi_id]['raw'], self.data[roi_id]['unmixed']]

        write_basic_json(out_fname, local_data, clobber=self.clobber)


class ValidCTH5Writer(OutputWriter):
    def run(self):
        out_fname = os.path.join(self.cache_dir,
                                 '%d_%d_valid_ct.h5' % (self.signal_plane.experiment_id,
                                                        self.crosstalk_plane.experiment_id))
        write_to_h5(out_fname, self.data, clobber=self.clobber)


class ValidCTH5WriterOld(OldOutputWriter):
    def run(self):
        out_dir = self.get_sub_dir('roi')
        out_fname = os.path.join(out_dir, '%d_valid_ct.json' % self.signal_plane.experiment_id)
        local_data = {}
        for roi_id in self.data:
            local_data[roi_id] = self.data[roi_id]['is_a_cell']
        write_basic_json(out_fname, local_data, clobber=self.clobber)
