from typing import List, Tuple
import numpy as np


class BasicDictWrapper(object):

    def __contains__(self, key) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        raise NotImplementedError("Did not implement iterator over "
                                  "BasicDictWrapper")

# classes for handling pure traces #############


class ROIChannels(BasicDictWrapper):
    """
    A wrapper around a dict meant to store all of the decrosstalk-related
    data for one ROI.

    Valid key, value pairs are:

    'signal' -- a 1-D np.ndarray of floats containing a trace
    'crosstalk' -- a 1-D np.ndarray of floats containing a trace
    'mixing_matrix' -- a 2x2 np.ndarray of floats
    'poorly_converged_signal' -- same type as 'signal'
    'poorly_converged_crosstalk' -- same type as 'crosstalk'
    'poorly_converged_mixing_matrix' -- same type as 'mixing_matrix'
    'use_avg_mixing_matrix' -- a boolean
    'unclipped_signal' -- same as 'signal'
    """

    def __init__(self):
        self._data = {}
        self._valid_keys = ('signal',
                            'crosstalk',
                            'mixing_matrix',
                            'poorly_converged_signal',
                            'poorly_converged_crosstalk',
                            'poorly_converged_mixing_matrix',
                            'use_avg_mixing_matrix',
                            'unclipped_signal')

    def _validate_np_array(self,
                           key: str,
                           value: np.ndarray,
                           expected_ndim: int = None,
                           expected_shape: Tuple = None,
                           expected_dtype: np.dtype = None) -> bool:

        if not isinstance(value, np.ndarray):
            msg = '%s must be a np.ndarray; you gave %s\n' % (key, type(value))
            raise ValueError(msg)
        elif not len(value.shape) == expected_ndim:
            msg = '%s must be a ' % key
            msg += '%s-dimensional array; ' % str(expected_ndim)
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

    def __setitem__(self, key: str, value: np.ndarray):
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
        elif key == 'unclipped_signal':
            self._validate_np_array('unclipped_signal', value,
                                    expected_ndim=1,
                                    expected_shape=None,
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

    def __getitem__(self, key: str) -> np.ndarray:
        if key in self._valid_keys:
            return np.copy(self._data[key])
        msg = 'Keys for ROIChannels must be one of:\n'
        msg += str(self._valid_keys)
        msg += '\nyou gave: %s' % key
        raise KeyError(msg)

    def pop(self, key):
        raise NotImplementedError("ROIChannels does not support pop")

    def keys(self) -> List[str]:
        return list(self._data.keys())


class ROIDict(BasicDictWrapper):
    """
    A wrapper around a dict meant to store the decrosstalk-related
    data for many ROIs.

    Key/Value pairs must all be (int, ROIChannels) where int is the
    ROI's roi_id.
    """

    def __init__(self):
        self._data = {}

    def __setitem__(self, roi_id: int, value: ROIChannels):
        if not isinstance(roi_id, int):
            msg = 'ROIDict keys must be ints; '
            msg += 'you are using %s' % str(type(roi_id))
            raise KeyError(msg)
        if not isinstance(value, ROIChannels):
            msg = 'ROIDict values must be ROIChannels; '
            msg += 'you are using %s' % (str(type(value)))
            raise ValueError(msg)
        self._data[roi_id] = value

    def __getitem__(self, roi_id: int) -> ROIChannels:
        return self._data[roi_id]

    def pop(self, key: int) -> ROIChannels:
        return self._data.pop(key)

    def keys(self) -> List[int]:
        return list(self._data.keys())


class ROISetDict(object):
    """
    Wrapper around dict meant to store the ROI and Neuropil trace
    data for a set of ROIs.

    Valid keys are 'roi' and 'neuropil', each of which stores an
    ROIDict containing the relevant data for the ROIs measured in
    that type of footprint (ROI or neuropil).
    """

    def __init__(self):
        self._roi = ROIDict()
        self._neuropil = ROIDict()

    def __setitem__(self, key, value):
        raise NotImplementedError("ROISetDict does not implement __setitem__")

    def __getitem__(self, key: str) -> ROIDict:
        if key == 'roi':
            return self._roi
        elif key == 'neuropil':
            return self._neuropil
        msg = "The only legal keys for ROISetDict "
        msg += "are 'roi' and 'neuropil'; "
        msg += "you passed '%s'" % str(key)
        raise KeyError(msg)

    def pop(key):
        raise NotImplementedError("ROISetDict does not implement pop")

# classes for handling events ###################


class ROIEvents(object):
    """
    A wrapper around dict meant to store the active trace data for an
    ROI. Valid key/value pairs are

    'trace' -- a 1-D np.ndarray of floats containing trace values
    'events' -- a 1-D np.ndarray of ints containing indexes of active timesteps

    The idea is that, given a raw_trace stored somewhere else,

    raw_trace[ROIEvents['events']] == ROIEvents['trace']
    """

    def __init__(self):
        self._data = {}

    def __setitem__(self, key: str, value: np.ndarray):
        if not isinstance(value, np.ndarray):
            msg = "Can only store np.ndarrays in ROIEvents; "
            msg += "you passed in %s" % type(value)
            raise ValueError(msg)

        if len(value.shape) != 1:
            msg = "Can only store 1-dimensional "
            msg += "np.ndarrays in ROIEvents; "
            msg += "you passed in an "
            msg += "%d-dimensional array" % len(value.shape)
            raise ValueError(msg)

        if key == 'trace':
            if value.dtype != np.dtype(float):
                msg = "Traces must be floats in ROIEvents; "
                msg += "you passed in %s" % str(value.dtype)
                raise ValueError(msg)
            self._data[key] = value
        elif key == 'events':
            if value.dtype != np.dtype(int):
                msg = "Events must be ints in ROIEvents; "
                msg += "you passed in %s" % str(value.dtype)
                raise ValueError(msg)
            self._data[key] = value
        else:
            msg = "Only valid keys for ROIEvents are "
            msg += "'trace' and 'events'; you passed in "
            msg += "'%s'" % str(key)
            raise KeyError(msg)

    def __getitem__(self, key: str) -> np.ndarray:
        if key not in ('trace', 'events'):
            msg = "Only valid keys for ROIEvents are "
            msg += "'trace' and 'events'; you passed in "
            msg += "'%s'" % str(key)
            raise KeyError(msg)
        return np.copy(self._data[key])

    def pop(self, key: str) -> np.ndarray:
        return self._data.pop(key)


class ROIEventChannels(object):
    """
    A wrapper around dict meant to store the active trace data for the
    signal and crosstalk channels in an ROI.

    Valid keys are 'signal' and 'crosstalk', each of which stores an
    ROIEvents
    """

    def __init__(self):
        self._data = {}
        self._valid_keys = set(['signal', 'crosstalk'])

    def __setitem__(self, key: str, value: ROIEvents):
        if key not in self._valid_keys:
            msg = 'Only allowed keys in ROIEventChannels are:\n'
            msg += str(self._valid_keys)
            msg += '\nyou passed: %s' % str(key)
            raise KeyError(msg)
        if not isinstance(value, ROIEvents):
            msg = "ROIEvents are the only valid values for ROIEventChannels; "
            msg += "you passed %s" % str(type(value))
            raise ValueError(msg)
        self._data[key] = value

    def __getitem__(self, key: str) -> ROIEvents:
        if key not in self._valid_keys:
            msg = 'Only allowed keys in ROIEventChannels are:\n'
            msg += str(self._valid_keys)
            msg += '\nyou passed: %s' % str(key)
            raise KeyError(msg)
        return self._data[key]

    def pop(self, key: int) -> ROIEvents:
        return self._data.pop(key)


class ROIEventSet(BasicDictWrapper):
    """
    A wrapper around dict meant to store all of the ROIEventChannels for
    a set of ROIs. Valid keys are ints (the roi_ids of the ROIs). Valid
    values are ROIEventChannels.
    """

    def __init__(self):
        self._data = {}

    def __setitem__(self, key: int, value: ROIEventChannels):
        if not isinstance(key, int):
            msg = "Ints are the only valid keys for ROIEventSet; "
            msg += "you passed in %s" % str(type(key))
            raise KeyError(msg)
        if not isinstance(value, ROIEventChannels):
            msg = "ROIEventChannesl are the only valid values "
            msg += "for ROIEventSet; you passed in "
            msg += "%s" % str(type(value))
            raise ValueError(msg)
        self._data[key] = value

    def __getitem__(self, key: int) -> ROIEventChannels:
        if not isinstance(key, int):
            msg = "Ints are the only valid keys for ROIEventSet; "
            msg += "you passed in %s" % str(type(key))
            raise KeyError(msg)
        return self._data[key]

    def pop(self, key: int) -> ROIEventChannels:
        return self._data.pop(key)

    def keys(self) -> List[int]:
        return list(self._data.keys())
