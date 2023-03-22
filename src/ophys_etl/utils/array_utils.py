from typing import Optional
import h5py
import numpy as np
from typing import Union, Tuple


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
        upper_cutoff: Optional[float] = None,
        dtype: type = np.uint8) -> np.ndarray:
    """
    Normalize an array into an integer type with
    cutoff values

    Parameters
    ----------
    array: numpy.ndarray (float)
        array to be normalized
    lower_cutoff: Optional[float]
        threshold, below which will be = dtype.min
        (if None, will be set to array.min())
    upper_cutoff: Optional[float]
        threshold, above which will be = dtype.max
        (if None, will be set to array.max())
    dtype: type
        The type (must be a numpy integer type)
        to which to cast the array. The array
        will be renormalized so that it's dynamic
        range spans [np.iinfo(dtype).min, np.iinfo(dytpe).max]

    Returns
    -------
    normalized: numpy.ndarray
        normalized array of the specified integer type

    """
    final_max = np.iinfo(dtype).max
    final_min = np.iinfo(dtype).min

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
    normalized = normalized/delta
    normalized *= (final_max-final_min)
    normalized = np.round(normalized)
    normalized += final_min
    normalized = normalized.astype(dtype)
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

    scaled_array = normalize_array(
                        array=input_array,
                        lower_cutoff=lower_cutoff,
                        upper_cutoff=upper_cutoff)

    output_array = np.stack([scaled_array,
                             scaled_array,
                             scaled_array]).transpose(1, 2, 0)

    return output_array


def get_cutout_indices(
    center_dim: int,
    image_dim: int,
    cutout_dim: int,
) -> Tuple[int, int]:
    """Find the min/max indices of a cutout within the image size.

    Parameters
    ----------
    center_dim : int
        Center pixel coordinate in the dimension of interest.
    image_dim : int
        Image dimension size in pixels.
    cutout_dim : int
        Size of the dimension of the cutout in pixels.

    Returns
    -------
    cutout_indices : Tuple[int, int]
        Indices in the cutout to that cover the ROI in one dimension.
    """
    # Get size of cutout.
    lowside = max(0, center_dim - cutout_dim // 2)
    highside = min(image_dim, center_dim + cutout_dim // 2)
    return (lowside, highside)


def get_cutout_padding(dim_center: int,
                       image_dim_size: int,
                       cutout_dim: int) -> Tuple[int, int]:
    """If the requested cutout size is beyond any dimension of the image,
    found how much we need to pad by.

    Parameters
    ----------
    dim_center : int
        Index of the center of the ROI bbox in one of the image dimensions
        (row, col)
    image_dim_size : int
        Size of the image in the dimension we are testing for padding.
    cutout_dim : int
        Size of the dimension of the cutout.

    Returns
    -------
    padding : Tuple[int, int]
        Amount to pad on at the beginning and/or end of the cutout.
    """
    # If the difference between center and cutout size is less than zero,
    # we need to pad.
    lowside_pad = np.abs(
        min(0, dim_center - cutout_dim // 2))
    # If the difference between the center plus the cutout size is
    # bigger than the image size, we need to pad.
    highside_pad = max(
        0, dim_center + cutout_dim // 2 - image_dim_size)
    return (lowside_pad, highside_pad)
