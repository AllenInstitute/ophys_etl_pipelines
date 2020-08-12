from typing import List

from scipy.sparse import coo_matrix


def filter_longest_edge_length(coo_matrices: List[coo_matrix],
                               edge_threshold: int) -> List[coo_matrix]:
    """
    Low pass filters list of coo_matrices by the longest distance in the
    height or width (whichever is greater) against a user defined threshold.
    Matrix dimensions must be lower than edge_threshold, not equal to, to pass
    through the filter.
    Parameters
    ----------
    coo_matrices:
        The list of coo_matrices to low pass filter in both height and width
    edge_threshold:
        The threshold that both height and width of the coo_matrix must be
        under

    Returns
    -------
    List[coo_matrix]:
        The coo_matrices that had both length and width lower than the
        specified edge_threshold

    """
    filtered_rois = []
    for coo_roi in coo_matrices:
        max_row_or_col = max(coo_roi.col.ptp(),
                             coo_roi.row.ptp())
        # peak to peak gets actual length - 1
        if (max_row_or_col + 1) < edge_threshold:
            filtered_rois.append(coo_roi)
    return filtered_rois
