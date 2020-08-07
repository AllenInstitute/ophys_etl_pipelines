from typing import List

from scipy.sparse import coo_matrix


def filter_rois_by_longest_edge_length(coo_rois: List[coo_matrix],
                                       longest_edge_threshold: int) -> \
        List[coo_matrix]:
    filtered_rois = []
    for coo_roi in coo_rois:
        # check if s2p border artifact
        max_linear_dimension = max(coo_roi.col.ptp(),
                                   coo_roi.row.ptp())
        if max_linear_dimension < longest_edge_threshold:
            filtered_rois.append(coo_roi)
    return filtered_rois
