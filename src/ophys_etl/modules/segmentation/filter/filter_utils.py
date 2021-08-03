from typing import Optional
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.utils.roi_utils import (
    select_window_from_background)


def mean_metric_from_roi(
        roi: OphysROI,
        img: np.ndarray) -> float:
    """
    Calculate the mean metric value in an ROI based on
    a provided metric image

    Parameters
    ----------
    roi: OphysROI

    img: np.ndarray
        The metric image whose average value in the ROI is
        being returned. Shape is (nrows, ncolumns)

    Returns
    -------
    avg_metric_value: float
        The mean value of pixels in img that are a part
        of the ROI
    """

    rows = roi.global_pixel_array[:, 0]
    cols = roi.global_pixel_array[:, 1]
    return np.mean(img[rows, cols])


def median_metric_from_roi(
        roi: OphysROI,
        img: np.ndarray) -> float:
    """
    Calculate the median metric value in an ROI based on
    a provided metric image

    Parameters
    ----------
    roi: OphysROI

    img: np.ndarray
        The metric image whose average value in the ROI is
        being returned. Shape is (nrows, ncolumns)

    Returns
    -------
    avg_metric_value: float
        The median value of pixels in img that are a part
        of the ROI
    """

    rows = roi.global_pixel_array[:, 0]
    cols = roi.global_pixel_array[:, 1]
    return np.median(img[rows, cols])


def z_vs_background_from_roi(
        roi: OphysROI,
        img: np.ndarray,
        background_mask: np.ndarray,
        n_desired_background: Optional[int] = None) -> float:
    """
    For an ROI and a metric image, calculate the z-score
    of the average metric value in the ROI with respect to
    its local background, i.e.

    (roi_pixel_mean-background_pixel_mean)/background_pixel_std

    Parameters
    ----------
    roi: OphysROI

    img: np.ndarray
        The metric image whose average value in the ROI is
        being returned. Shape is (nrows, ncolumns)

    background_mask:
        A np.ndarray of booleans the same shape as
        img. Marked as False for any pixel that is included
        in an ROI. This is used when selecting background
        pixels to compare against. No pixels that are included
        in any ROI will be selected.

    n_desired_background: Optional[int]
        The minimum number of background pixels to use
        in calculating the z_score. If None, take roi.area.
        Default: None.

    Returns
    -------
    z_score: float
    """

    if img.shape != background_mask.shape:
        msg = f"img.shape: {img.shape}\n"
        msg += f"background_mask.shape: {background_mask.shape}\n"
        msg += "These must be equal"
        raise RuntimeError(msg)

    if n_desired_background is None:
        n_desired_background = roi.area

    # find a square window centered on the ROI that contains
    # the desired number of background pixels

    ((rowmin, rowmax),
     (colmin, colmax)) = select_window_from_background(
                             roi,
                             background_mask,
                             n_desired_background)

    local_mask = background_mask[rowmin:rowmax, colmin:colmax]
    local_img = img[rowmin:rowmax, colmin:colmax]
    background_pixels = local_img[local_mask].flatten()

    roi_rows = roi.global_pixel_array[:, 0]
    roi_cols = roi.global_pixel_array[:, 1]
    roi_pixels = img[roi_rows, roi_cols].flatten()

    roi_mean = np.mean(roi_pixels)
    background_mean = np.mean(background_pixels)

    # use interquartile range to estimate standard deviation
    # of background pixels
    q25, q75 = np.quantile(background_pixels, (0.25, 0.75))
    background_std = max(1.0e-6, (q75-q25)/1.34896)

    z_score = (roi_mean-background_mean)/background_std
    return z_score
