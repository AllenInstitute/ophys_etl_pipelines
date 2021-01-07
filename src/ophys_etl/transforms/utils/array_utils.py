import math
import logging
from typing import Tuple, Type, Union, List
import numpy as np
from scipy.sparse import coo_matrix
import h5py


def content_boundary_2d(arr: Union[np.ndarray, coo_matrix]) -> Tuple:
    """
    Get the minimal row/column boundaries for content in a 2d array.

    Parameters
    ==========
    arr: A 2d np.ndarray or coo_matrix. Other formats are also possible
        (csc_matrix, etc.) as long as they have the toarray() method to
         convert them to np.ndarray.

    Returns
    =======
    4-tuple of row/column boundaries that define the minimal rectangle
    around nonzero array content. Note that the maximum side boundaries
    follow python indexing rules, so the value returned is the actual
    max index + 1:
        top_bound: smallest row index
        bot_bound: largest row index + 1
        left_bound: smallest column index
        right_bound: largest column index + 1
    """
    if isinstance(arr, coo_matrix):
        col = arr.col
        row = arr.row
    else:
        if not isinstance(arr, np.ndarray):
            arr = arr.toarray()
        row, col = np.nonzero(arr)
    if not row.size:
        logging.warning("No content found. Either array is empty or all "
                        "elements equal zero.")
        return 0, 0, 0, 0
    left_bound = col.min()
    right_bound = col.max() + 1
    top_bound = row.min()
    bot_bound = row.max() + 1
    return top_bound, bot_bound, left_bound, right_bound


def content_extents(
        arr: Union[np.ndarray, coo_matrix],
        shape: Tuple[int, int],
        target_shape: Tuple[int, int] = None):
    """return the bounding box of size shape. Intended to be applied to
    a target data source, for example a set of video frames.

    Parameters
    ----------
    arr: A 2d np.ndarray or coo_matrix. Other formats are also possible
        (csc_matrix, etc.) as long as they have the toarray() method to
         convert them to np.ndarray.
    shape: (Tuple[int,int]) Desired final shape after padding is applied.
        If smaller than the input array, will return the input array
        without any changes.
    target_shape: (Tuple[int, int]) Extent of array to be indexed. If None
        padding will still handle top and left, but bottom and right padding
        will always be zero

    Returns
    -------
    indexing_bounds: tuple
        4-tuple of row/column boundaries
    pad_width: tuple(tuple, tuple)
        to be passed into numpy.pad as pad_width

    """
    boundaries = content_boundary_2d(arr)
    height = boundaries[1] - boundaries[0]
    width = boundaries[3] - boundaries[2]

    vertical_pad = shape[0] - height
    horizontal_pad = shape[1] - width

    if vertical_pad % 2 == 0:
        top_pad = bottom_pad = int(vertical_pad / 2)
    else:
        top_pad = math.floor(vertical_pad / 2)
        bottom_pad = top_pad + 1
    if horizontal_pad % 2 == 0:
        left_pad = right_pad = int(horizontal_pad / 2)
    else:
        left_pad = math.floor(horizontal_pad / 2)
        right_pad = left_pad + 1

    top = boundaries[0] - top_pad
    bot = boundaries[1] + bottom_pad
    left = boundaries[2] - left_pad
    right = boundaries[3] + right_pad

    pad_width = [[0, 0], [0, 0]]
    if top < 0:
        pad_width[0][0] = abs(top)
        top = 0
    if left < 0:
        pad_width[1][0] = abs(left)
        left = 0
    if target_shape is not None:
        dh = target_shape[0] - bot
        if dh < 0:
            pad_width[0][1] = abs(dh)
            bot = target_shape[0]
        dw = target_shape[1] - right
        if dw < 0:
            pad_width[1][1] = abs(dw)
            right = target_shape[1]

    indexing_bounds = (top, bot, left, right)
    pad_width = tuple([tuple(i) for i in pad_width])

    return indexing_bounds, pad_width


def crop_2d_array(arr: Union[np.ndarray, coo_matrix]) -> np.ndarray:
    """
    Crop a 2d array to a rectangle surrounding all nonzero elements.

    Parameters
    ==========
    arr: A 2d np.ndarray or coo_matrix. Other formats are also possible
        (csc_matrix, etc.) as long as they have the toarray() method to
         convert them to np.ndarray.

    Raises
    ======
    ValueError if all elements are nonzero.
    """
    boundaries = content_boundary_2d(arr)
    if not isinstance(arr, np.ndarray):
        arr = arr.toarray()
    if sum(boundaries) == 0:
        raise ValueError("Cannot crop an empty array, or an array where all "
                         "elements are zero.")
    top_bound, bot_bound, left_bound, right_bound = boundaries
    return arr[top_bound:bot_bound, left_bound:right_bound]


def center_pad_2d(arr: np.ndarray, shape: Tuple[int, int],
                  value: Type[np.dtype] = 0,
                  allow_overflow: bool = True) -> Union[None, np.ndarray]:
    """
    Add padding around a numpy array such that the original array data
    stays in the center. If padding cannot be evenly applied due to the
    size of array data and desired shape, then the extra padding will
    be applied after (on the bottom for the row axis and right side
    for the column axis).

    Parameters
    ==========
    arr: (np.ndarray) 2d array of data
    shape: (Tuple[int,int]) Desired final shape after padding is applied.
        If smaller than the input array, will return the input array
        without any changes.
    value: (inherit from np.dtype) Any valid numpy dtype value. Should
        be homogenous with the input array.
    allow_overflow: (bool) If true, will return array unchanged if the
        array size is larger than the value for `shape`. If false,
        will return None instead.
    """
    if arr.size == 0:
        return np.full(shape, value)
    img_size = arr.shape
    vertical_pad = shape[0] - img_size[0]
    horizontal_pad = shape[1] - img_size[1]

    if (img_size[0] > shape[0]) or (img_size[1] > shape[1]):
        logging.warning("Specified shape after padding is too small. "
                        "Returning input array without padding.")
        if allow_overflow:
            return arr
        else:
            return None

    if vertical_pad % 2 == 0:
        top_pad = bottom_pad = int(vertical_pad / 2)
    else:
        top_pad = math.floor(vertical_pad / 2)
        bottom_pad = top_pad + 1
    if horizontal_pad % 2 == 0:
        left_pad = right_pad = int(horizontal_pad / 2)
    else:
        left_pad = math.floor(horizontal_pad / 2)
        right_pad = left_pad + 1

    return np.pad(arr, ((top_pad, bottom_pad), (left_pad, right_pad)),
                  mode="constant", constant_values=(value,))


def downsample_array(
        array: Union[h5py.Dataset, np.ndarray],
        input_fps: int = 31,
        output_fps: int = 4,
        strategy: str = 'average',
        random_seed: int = 0) -> np.ndarray:
    """Downsamples an array-like object along axis=0

    Parameters
    ----------
        array: h5py.Dataset or numpy.ndarray
            the input array
        input_fps: int
            frames-per-second of the input array
        output_fps: int
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
        array: np.ndarray, quantiles: List, calc_cutoffs_from_array: Union[None, np.ndarray] = None) -> np.ndarray:
    """Clips values > upper_cutoff or < lower_cutoff
    Normalizes all pixel values to be between [0, 255]

    Parameters
    ----------
    array: numpy.ndarray (float)
        array to be clipped
    quantiles: list
        quantiles to use to calculate cutoffs
    calc_cutoffs_from_array: numpy.ndarray (float)
        calculate cutoffs using this array

    Returns
    -------
    array: numpy.ndarray
        normalized array

    """
    reference_array = calc_cutoffs_from_array if calc_cutoffs_from_array is not None else array
    reference_array = reference_array.flatten()

    lower_cutoff, upper_cutoff = np.quantile(reference_array, quantiles)
    array = np.clip(array, a_min=lower_cutoff, a_max=upper_cutoff)

    arr_min = array.min()
    arr_max = array.max()
    array = (array - arr_min) / (arr_max - arr_min)
    array *= 255
    array = array.astype('uint8')
    return array