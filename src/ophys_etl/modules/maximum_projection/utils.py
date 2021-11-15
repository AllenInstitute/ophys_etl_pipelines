import numpy as np
from itertools import product


def reflect_indexes(
        raw_indexes: np.ndarray,
        dim: int) -> np.ndarray:
    """
    Take an array of indexes that may go out of bounds for an array's
    shape and cast them into valid values by reflecting them about
    the bounds of the array.

    Parameters
    ----------
    raw_indexes: np.ndarray

    dim: int
        Size of array dimension raw_indexes is meant to refer to

    Returns
    -------
    reflected_indexes: np.ndarray
        Values of raw_index that exceed dim will be reflected about dim
        (i.e. if dim = 9 and raw_index = [9, 10, 11] then reflected index
        will be [7, 6, 5]; if raw_index = [-1, -2, -3], reflected_index
        will be [1, 2, 3])

    Note
    ----
    Acts on raw_indexes in place
    """

    if raw_indexes.max() < dim and raw_indexes.min() >= 0:
        return raw_indexes

    raw_indexes[(raw_indexes < 0)] *= -1
    too_big = (raw_indexes >= dim)
    raw_indexes[too_big] *= -1
    raw_indexes[too_big] += 2*(dim-1)

    return reflect_indexes(raw_indexes, dim)


def median_filter(
        img: np.ndarray,
        kernel_size: int) -> np.ndarray:
    """
    Apply a median filter to a single frame in a video.

    Parameters
    ----------
    img: np.ndarray
        The 2D image being filtered

    kernel_size:int
        The size length of the square kernel used to find
        the median value about each pixel

    Returns
    -------
    filtered_img: np.ndarray
    """
    img_shape = img.shape
    img = img.flatten()

    pixel_indexes_1d = np.arange(len(img), dtype=int)
    pixel_indexes_2d = np.unravel_index(pixel_indexes_1d,
                                        img_shape)

    k0 = -kernel_size//2
    kernel_1d = np.arange(k0, k0+kernel_size, 1)

    neighbor_array = np.zeros((len(pixel_indexes_1d), kernel_size**2),
                              dtype=float)

    for i_neighbor, (drow, dcol) in enumerate(product(kernel_1d, kernel_1d)):
        neighbor_row = reflect_indexes(pixel_indexes_2d[0]+drow,
                                       img_shape[0])
        neighbor_col = reflect_indexes(pixel_indexes_2d[1]+dcol,
                                       img_shape[1])

        neighbor_1d = np.ravel_multi_index([neighbor_row,
                                            neighbor_col],
                                           img_shape)

        neighbor_array[:, i_neighbor] = img[neighbor_1d]
    return np.median(neighbor_array, axis=1).reshape(img_shape)
