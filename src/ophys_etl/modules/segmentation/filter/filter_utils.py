import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI


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
