import numpy as np


class BasicDictWrapper(object):

    def pop(self, key):
        return self._data.pop(key)

    def __contains__(self, key):
        if key in self._data:
            return True
        return False

    def keys(self):
        return list(self._data.keys())

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        raise NotImplementedError("Did not implement iterator over "
                                  "BasicDictWrapper")

class ROIChannels(BasicDictWrapper):

    def __init__(self):
        self._data = {}
        self._valid_keys = ('signal',
                            'crosstalk',
                            'mixing_matrix',
                            'poorly_converged_signal',
                            'poorly_converged_crosstalk',
                            'poorly_converged_mixing_matrix',
                            'use_avg_mixing_matrix')

    def _validate_np_array(self, key, value,
                           expected_ndim=None,
                           expected_shape=None,
                           expected_dtype=None):

        if not isinstance(value, np.ndarray):
            msg = '%s must be a np.ndarray; you gave %s\n' % (key, type(value))
            raise ValueError(msg)
        elif not len(value.shape) == expected_ndim:
            msg = '%s must be a %d-dimensional array; ' % (key, expected_ndim)
            msg += 'you gave %d-dimensional array' % (len(value.shape))
            raise ValueError(msg)
        elif value.dtype != expected_dtype:
            msg = '%s must be a np.ndarray ' % key
            msg += 'of dtype %s;' % str(expected_dtype)
            msg += ' you gave %s' % str(value.dtype)
            raise ValueError(msg)
        elif expected_shape is not None and value.shape != expected_shape:
            msg = '%s must be of shape %s; ' % (key, str(expected_shape))
            msg += 'you gave %s' % (str(value.shape))
            raise ValueError(msg)
        return True

    def __setitem__(self, key: str, value):
        float_type = np.dtype(float)
        if key == 'signal':
            self._validate_np_array('signal', value,
                                    expected_ndim=1,
                                    expected_shape=None,
                                    expected_dtype=float_type)
            self._data[key] = value
        elif key == 'crosstalk':
            self._validate_np_array('crosstalk', value,
                                    expected_ndim=1,
                                    expected_shape=None,
                                    expected_dtype=float_type)
            self._data[key] = value
        elif key == 'mixing_matrix':
            self._validate_np_array('mixing_matrix', value,
                                    expected_ndim=2,
                                    expected_shape=(2, 2),
                                    expected_dtype=float_type)
            self._data[key] = value
        elif key == 'poorly_converged_signal':
            self._validate_np_array('poorly_converged_signal', value,
                                    expected_ndim=1,
                                    expected_shape=None,
                                    expected_dtype=float_type)
            self._data[key] = value
        elif key == 'poorly_converged_crosstalk':
            self._validate_np_array('poorly_converged_crosstalk', value,
                                    expected_ndim=1,
                                    expected_shape=None,
                                    expected_dtype=float_type)
            self._data[key] = value
        elif key == 'poorly_converged_mixing_matrix':
            self._validate_np_array('poorly_converged_mixing_matrix', value,
                                    expected_ndim=2,
                                    expected_shape=(2, 2),
                                    expected_dtype=float_type)
            self._data[key] = value
        elif key == 'use_avg_mixing_matrix':
            if not isinstance(value, bool):
                msg = 'use_avg_mixing_matrix must be a boolean; '
                msg += 'you gave %s' % type(value)
                raise ValueError(msg)
            self._data[key] = value
        else:
            msg = 'Keys for ROIChannels must be one of:\n'
            msg += str(self._valid_keys)
            msg += '\nyou gave: %s' % key
            raise KeyError(msg)

    def __getitem__(self, key: str):
        if key in self._valid_keys:
            return np.copy(self._data[key])
        msg = 'Keys for ROIChannels must be one of:\n'
        msg += str(self._valid_keys)
        msg += '\nyou gave: %s' % key
        raise KeyError(msg)

    def pop(self, key):
        raise NotImplementedError("ROIChannels does not support pop")


class ROIDict(BasicDictWrapper):

    def __init__(self):
        self._data = {}

    def __setitem__(self, roi_id: int, value: ROIChannels):
        if not isinstance(roi_id, int):
            msg = 'ROIDict keys must be ints; '
            msg += 'you are using %s' % str(type(roi_id))
            raise KeyError(msg)
        if not isinstance(value, ROIChannels):
            msg = 'ROIDict values must be ROIChannels; '
            msg += 'youa reusing %s' % (str(type(value)))
            raise ValueError(msg)
        self._data[roi_id] = value

    def __getitem__(self, roi_id: int):
        return self._data[roi_id]


class ROISetDict(object):

    def __init__(self):
        self._roi = ROIDict()
        self._neuropil = ROIDict()

    def __setitem__(self, key, value):
        raise NotImplementedError("ROISetDict does not implement __setitem__")

    def __getitem__(self, key: str):
        if key == 'roi':
            return self._roi
        elif key == 'neuropil':
            return self._neuropil
        msg = "The only legal keys for ROISetDict "
        msg += "are 'roi' and 'neuropil'; "
        msg += "you passed '%s'" % str(key)
        raise KeyError(msg)
