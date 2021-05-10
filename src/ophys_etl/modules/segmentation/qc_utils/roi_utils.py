from typing import List, Tuple
import numpy as np
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI


def add_roi_boundaries_to_img(img: np.ndarray,
                              roi_list: List[OphysROI],
                              color: Tuple[int] = (255, 0, 0),
                              alpha: float = 0.25) -> np.ndarray:
    """
    Add colored ROI boundaries to an image

    Parameters
    ----------
    img: np.ndarray
        RGB representation of image

    roi_list: List[OphysROI]
        list of ROIs to add to image

    color: Tuple[int]
        color of ROI border as RGB tuple (default: (255, 0, 0))

    alpha: float
        transparency factor to apply to ROI (default=0.25)

    Returns
    -------
    new_img: np.ndarray
        New image with ROI borders superimposed
    """

    new_img = np.copy(img)
    for roi in roi_list:
        bdry = roi.boundary_mask
        for icol in range(roi.width):
            for irow in range(roi.height):
                if not bdry[irow, icol]:
                    continue
                yy = roi.y0 + irow
                xx = roi.x0 + icol
                for ic in range(3):
                    old_val = np.round(img[yy, xx, ic]*(1.0-alpha)).astype(int)
                    new_img[yy, xx, ic] = old_val
                    new_val = np.round(alpha*color[ic]).astype(int)
                    new_img[yy, xx, ic] += new_val

    new_img = np.where(new_img <= 255, new_img, 255)
    return new_img
