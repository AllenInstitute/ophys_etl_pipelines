from typing import List

from scipy.sparse import coo_matrix


def filter_by_aspect_ratio(coo_matrices: List[coo_matrix],
                           aspect_threshold: float) -> List[coo_matrix]:
    """
    Returns a list where ROIs with aspect ratio <= aspect_threshold are
    removed. Aspect ratio is min(heigh/width, width/height)
    Parameters
    ----------
    coo_matrices:
        The list of coo_matrices to low pass filter in both height and width
    aspect_threshold:
        The inclusive threshold below which ROIs will be removed

    Returns
    -------
    List[coo_matrix]:
        The coo_matrices that are not removed by this filter

    """
    filtered_rois = []
    for coo_roi in coo_matrices:
        height = coo_roi.row.ptp() + 1
        width = coo_roi.col.ptp() + 1
        ratio = min(height/width, width/height)
        if ratio > aspect_threshold:
            filtered_rois.append(coo_roi)
    return filtered_rois
