from typing import Tuple, List, Union, Dict
import numpy as np
from ophys_etl.utils.array_utils import (
    array_to_rgb,
    normalize_array)

from ophys_etl.types import OphysROI
from ophys_etl.utils.rois import (
    sanitize_extract_roi_list,
    extract_roi_to_ophys_roi)


def plot_rois_over_img(
        img: np.ndarray,
        roi_list: Union[List[OphysROI], List[Dict]],
        color: Union[Tuple[int, int, int],
                     Dict[int, Tuple[int, int, int]]],
        alpha: float = 0.5) -> np.ndarray:
    """
    Plot contours from a list of ROIs over a provided image

    Parameters
    ----------
    img: np.ndarray
        The image, either grayscale or RGB

    roi_list: Union[List[OphysROI], List[Dict]]
        A list of ROIs represented either as an OphysROI
        or an ExtractROI

    color: Union[Tuple[int, int, int],
                 Dict[int, Tuple[int, int, int]]
        Either a tuple indicating a single RGB color for all ROIs
        or a dict mapping ROI ID to an RGB color (as a tuple of ints)

    alpha: float
        The transparency

    Returns
    -------
    new_img: np.ndarray
        An RGB image with the ROIs overplotted (does not
        modify img in place)
    """
    if len(img.shape) == 2:
        img = array_to_rgb(img)
    elif len(img.shape) == 3:
        if img.shape[2] != 3:
            msg = f"Cannot handle image with shape {img.shape}"
            raise ValueError(msg)
        if img.dtype != np.uint8:
            new_img = np.zeros(img.shape, dtype=np.uint8)
            upper_cutoff = img.max()
            lower_cutoff = img.min()
            for ic in range(3):
                new_img[:, :, ic] = normalize_array(
                                         img[:, :, ic],
                                         lower_cutoff=lower_cutoff,
                                         upper_cutoff=upper_cutoff)
            img = new_img
    else:
        msg = f"Cannot handle image with shape {img.shape}"
        raise ValueError(msg)

    new_img = add_list_of_roi_contours_to_img(
                img=img,
                roi_list=roi_list,
                color=color,
                alpha=alpha)
    return new_img


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


def add_list_of_roi_contours_to_img(
        img: np.ndarray,
        roi_list: Union[List[OphysROI], List[Dict]],
        color: Union[Tuple[int, int, int],
                     Dict[int, Tuple[int, int, int]]] = (255, 0, 0),
        alpha: float = 0.25) -> np.ndarray:
    """
    Add colored ROI contours to an image

    Parameters
    ----------
    img: np.ndarray
        RGB representation of image

    roi_list: List[OphysROI]
        list of ROIs to add to image

    color: Union[Tuple[int,int, int],
                 Dict[int, Tuple[int, int, int]]
        Either a representing an RGB color, or a dict
        mapping roi_id to tuples representing RGB colors
        (default = (255, 0, 0))

    alpha: float
        transparency factor to apply to ROI (default=0.25)

    Returns
    -------
    new_img: np.ndarray
        New image with ROI borders superimposed
    """

    new_img = np.copy(img)
    if len(roi_list) == 0:
        return new_img

    if not isinstance(roi_list[0], OphysROI):
        roi_list = sanitize_extract_roi_list(roi_list)
        roi_list = [extract_roi_to_ophys_roi(roi)
                    for roi in roi_list]

    if isinstance(color, tuple):
        color_map = {roi.roi_id: color for roi in roi_list}
    else:
        color_map = color

    for roi in roi_list:
        new_img = add_roi_contour_to_img(
                      new_img,
                      roi,
                      color_map[roi.roi_id],
                      alpha)

    return new_img
