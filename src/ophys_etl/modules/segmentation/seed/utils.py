import numpy as np
from typing import Set, Tuple
from skimage.morphology import square, binary_dilation
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


def dilated_coordinates(pixels: Set[Tuple[int, int]],
                        dilation_buffer: int) -> Set[Tuple[int, int]]:
    """binary dilation of a set of coordinates

    Parameters
    ----------
    pixels: Set[Tuple[int, int]]
        coordinates to dilate
    dilation_buffer: int
        in pixels, the buffer around the provided pixels to dilate

    Returns
    -------
    dilated_pixels: Set[Tuple[int, int]]
        the updated set of pixels

    """
    extents = np.array([i for i in pixels]).max(axis=0)
    shape = (extents[0] + 1 + dilation_buffer,
             extents[1] + 1 + dilation_buffer)

    # create an empty FOV with only specified pixels
    full_fov = np.zeros(shape, dtype=bool)
    rows, cols = np.array([i for i in pixels]).T
    full_fov[rows, cols] = True

    # retrieve set of dilated coordinates
    selem = square(2 * dilation_buffer + 1)
    full_fov = binary_dilation(full_fov, selem)
    dilated_pixels = {tuple(i) for i in np.argwhere(full_fov)}

    return dilated_pixels


class Seed(TypedDict):
    coordinates: Tuple[int, int]
    value: float
    exclusion_reason: str
