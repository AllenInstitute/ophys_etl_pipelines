import h5py
import numpy as np
from typing import Union


def downsample_array(
        array: Union[h5py.Dataset, np.ndarray],
        input_fps: float = 31.0,
        output_fps: float = 4.0,
        strategy: str = 'average',
        random_seed: int = 0) -> np.ndarray:
    """Downsamples an array-like object along axis=0

    Parameters
    ----------
        array: h5py.Dataset or numpy.ndarray
            the input array
        input_fps: float
            frames-per-second of the input array
        output_fps: float
            frames-per-second of the output array
        strategy: str
            downsampling strategy. 'random', 'maximum', 'average',
            'first', 'last'. Note 'maximum' is not defined for
            multi-dimensional arrays
        random_seed: int
            passed to numpy.random.default_rng if strategy is 'random'

    Returns:
        array_out: numpy.ndarray
            array downsampled along axis=0
    """
    if output_fps > input_fps:
        raise ValueError('Output FPS cannot be greater than input FPS')
    if (strategy == 'maximum') & (len(array.shape) > 1):
        raise ValueError("downsampling with strategy 'maximum' is not defined")

    npts_in = array.shape[0]
    npts_out = int(npts_in * output_fps / input_fps)
    bin_list = np.array_split(np.arange(npts_in), npts_out)

    array_out = np.zeros((npts_out, *array.shape[1:]))

    if strategy == 'random':
        rng = np.random.default_rng(random_seed)

    sampling_strategies = {
            'random': lambda arr, idx: arr[rng.choice(idx)],
            'maximum': lambda arr, idx: arr[idx].max(axis=0),
            'average': lambda arr, idx: arr[idx].mean(axis=0),
            'first': lambda arr, idx: arr[idx[0]],
            'last': lambda arr, idx: arr[idx[-1]]
            }

    sampler = sampling_strategies[strategy]
    for i, bin_indices in enumerate(bin_list):
        array_out[i] = sampler(array, bin_indices)

    return array_out


def normalize_array(
        array: np.ndarray, lower_cutoff: float,
        upper_cutoff: float) -> np.ndarray:
    """Normalize an array into uint8 with cutoff values

    Parameters
    ----------
    array: numpy.ndarray (float)
        array to be normalized
    lower_cutoff: float
        threshold, below which will be = 0
    upper_cutoff: float
        threshold, abovewhich will be = 255

    Returns
    -------
    normalized: numpy.ndarray (uint8)
        normalized array

    """
    normalized = np.copy(array)
    normalized[array < lower_cutoff] = lower_cutoff
    normalized[array > upper_cutoff] = upper_cutoff
    normalized -= lower_cutoff
    normalized = np.uint8(normalized * 255 / (upper_cutoff - lower_cutoff))
    return normalized


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """
    Calculate all of the pairwise distances between the rows
    in a np.ndarray

    Parameters
    ----------
    points: np.ndarray
        Shape is (n_points, n_dimensions)

    Returns
    -------
    distances: np.ndarray
        A (n_points, n_points) array. The i,jth element
        is the Euclidean distance between the ith and jth
        rows of the input points.

    Notes
    -----
    As n_points, n_dimensions approach a few thousand, this is
    several orders of magnitude faster than scipy.distances.cdist
    """
    p_dot_p = np.dot(points, points.T)
    dsq = np.zeros((points.shape[0], points.shape[0]), dtype=float)
    for ii in range(points.shape[0]):
        dsq[ii, :] += p_dot_p[ii, ii]
        dsq[:, ii] += p_dot_p[ii, ii]
        dsq[ii, :] -= 2.0*p_dot_p[ii, :]
    return np.sqrt(dsq)
