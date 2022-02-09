from typing import Optional
import h5py
import numpy as np
from typing import Union


def n_frames_from_hz(
        input_frame_rate: float,
        downsampled_frame_rate: float) -> int:
    """
    Find the number of frames to group together to downsample
    a video from input_frame_rate to downsampled_frame_rate

    Parameters
    ----------
    input_frame_rate: float

    downsampled_frame_rate: float

    Returns
    -------
    frames_to_group: int

    Notes
    -----
    If input_frame_rate/downsampled_frame_rate < 1, will return 1
    """

    frames_to_group = np.round(input_frame_rate/downsampled_frame_rate)
    frames_to_group = frames_to_group.astype(int)
    return max(1, frames_to_group)


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
    frames_to_group = n_frames_from_hz(input_fps, output_fps)
    npts_out = max(1, np.ceil(npts_in/frames_to_group).astype(int))

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
    for i_out, i0 in enumerate(range(0, npts_in, frames_to_group)):
        i1 = min(npts_in, i0+frames_to_group)
        array_out[i_out] = sampler(array,
                                   np.arange(i0, i1, dtype=int))

    return array_out


def normalize_array(
        array: np.ndarray,
        lower_cutoff: Optional[float] = None,
        upper_cutoff: Optional[float] = None) -> np.ndarray:
    """Normalize an array into uint8 with cutoff values

    Parameters
    ----------
    array: numpy.ndarray (float)
        array to be normalized
    lower_cutoff: Optional[float]
        threshold, below which will be = 0
        (if None, do not clip the array)
    upper_cutoff: Optional[float]
        threshold, abovewhich will be = 255
        (if None, do not clip the array)

    Returns
    -------
    normalized: numpy.ndarray (uint8)
        normalized array

    """
    normalized = np.copy(array).astype(float)
    if lower_cutoff is not None:
        normalized[array < lower_cutoff] = lower_cutoff
    else:
        lower_cutoff = normalized.min()

    if upper_cutoff is not None:
        normalized[array > upper_cutoff] = upper_cutoff
    else:
        upper_cutoff = normalized.max()

    normalized -= lower_cutoff

    delta = upper_cutoff-lower_cutoff

    normalized = np.round(normalized * 255 / delta)
    normalized = normalized.astype(np.uint8)
    return normalized


def array_to_rgb(
        input_array: np.ndarray,
        lower_cutoff: Optional[float] = None,
        upper_cutoff: Optional[float] = None) -> np.ndarray:
    """
    Take a 2-D np.ndarray of arbitrary dtype and cast it into
    a 3-D array of np.uint8 representing an RGB image

    Parameters
    ----------
    input_array: np.ndarray
        (nrows, ncols)

    lower_cutoff: Optional[float]
        threshold, below which will be = 0
        (if None, do not clip the array)

    upper_cutoff: Optional[float]
        threshold, abovewhich will be = 255
        (if None, do not clip the array)

    Returns
    -------
    rgb_array:
        (nrows, ncols, 3); original data clipped
        (if cutoffs set) and scaled to np.uint8
    """
    output_array = np.zeros((input_array.shape[0],
                             input_array.shape[1],
                             3), dtype=np.uint8)

    scaled_array = normalize_array(
                        array=input_array,
                        lower_cutoff=lower_cutoff,
                        upper_cutoff=upper_cutoff)

    for ic in range(3):
        output_array[:, :, ic] = scaled_array
    return output_array
