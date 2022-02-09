from typing import Tuple
import numpy as np
from ophys_etl.types import OphysROI


def add_roi_contour_to_img(
        img: np.ndarray,
        roi: OphysROI,
        color: Tuple[int, int, int],
        alpha: float) -> np.ndarray:
    """
    Add colored ROI contour to an image

    Parameters
    ----------
    img: np.ndarray
        RGB representation of image

    roi: OphysROI

    color: Tuple[int]
        RGB color of ROI

    alpha: float

    Returns
    -------
    img: np.ndarray

    Note
    ----
    While this function does return an image, it also operates
    on img in place
    """
    bdry = roi.contour_mask
    valid = np.argwhere(bdry)
    rows = np.array([r+roi.y0 for r in valid[:, 0]])
    cols = np.array([c+roi.x0 for c in valid[:, 1]])
    for ic in range(3):
        old_vals = img[rows, cols, ic]
        new_vals = np.round(alpha*color[ic]+(1.0-alpha)*old_vals).astype(int)
        img[rows, cols, ic] = new_vals
    img = np.where(img >= 255, 255, img)
    return img
