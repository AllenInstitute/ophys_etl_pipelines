from typing import Tuple
import numpy as np
import cv2


def add_scale(array: np.ndarray,
              scale_position: Tuple[int, int],
              um_per_pixel: float = 400 / 512,
              scale_size_um: float = 10,
              color: int = 0,
              thickness_um: float = 1.0,
              fontScale: float = 0.2) -> np.ndarray:
    """
    Adds a scale bar onto an array using opencv.

    Parameters
    ----------
    array: np.ndarray
        function only accepts np.uint8 2D arrays
    scale_position: Tuple
        image coordinates for the corner of the scale bars
    um_per_pixel: float = 400/512
        resolution of the video in microns per pixel
    scale_size_um: float
        length in microns of scale bars
    color: int
        default 0. passed as `color` to cv2.line and cv2.putText
        0 = black.
    thickness_um: float
        converted to pixels and passed as `thickness` to
        cv2.line and cv2.putText
    fontScale: float
        default 0.3. passed as `fontScale` to cv2.putText. See notes.

    Returns
    -------
    annotated: numpy.ndarray
        same shape and datatype as input

    Raises
    ------
    NotImplementedError
        if array is not a 2D uint8 np.ndarry

    Notes
    -----
    It is not obvious from opencv docs how many pixels in size text will be
    using cv2.putText. Empirically:

    >>> x = np.full((128, 128), 255, dtype='uint8')
    >>> cv2.putText(x, '10um', (5, 123), cv2.FONT_HERSHEY_SIMPLEX,
                0.3, 0, 1, cv2.LINE_4)
    >>> plt.imshow(x, cmap='gray', interpolation='nearest')

    produces text that is 7 pixels high. opencv provides a utility
    to check this:

    >>> cv2.getTextSize('10um', cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
    ((28, 7), 3)

    where the second element of the tuple is the height of the bounding
    box of the text. These results depend not only on fontScale but also
    thickness. For a line thickness of 1, some examples of (fontScale, height)
    are (0.3, 7), (0.7, 16), (1.0, 22)

    """
    if (array.ndim != 2) | (array.dtype != 'uint8'):
        raise NotImplementedError(
            "add_scale() only works for 2D arrays of type uint8. "
            f"provided array is {array.ndim}D and type {array.dtype}.")

    annotated = np.copy(array)
    length = np.round(scale_size_um / um_per_pixel).astype('int')
    thickness = np.round(thickness_um / um_per_pixel).astype('int')
    pt1 = (scale_position[0], scale_position[1] - length + 1)
    pt2 = (scale_position[0] + length - 1, scale_position[1])
    font = cv2.FONT_HERSHEY_SIMPLEX
    linetype = cv2.LINE_4
    cv2.putText(
            annotated, f"{int(scale_size_um)}um", pt2,
            font, fontScale, color, thickness, linetype)
    cv2.line(annotated, scale_position, pt1, color, thickness, linetype)
    cv2.line(annotated, scale_position, pt2, color, thickness, linetype)
    return annotated