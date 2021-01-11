import h5py
import os


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
