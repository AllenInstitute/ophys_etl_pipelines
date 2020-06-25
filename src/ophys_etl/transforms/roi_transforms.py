from typing import List, Optional, Tuple

import numpy as np
from scipy.sparse import coo_matrix


def suite2p_rois_to_coo(suite2p_stats: np.ndarray,
                        movie_shape: Tuple[int, int]) -> List[coo_matrix]:
    """Convert suite2p formatted rois to sparse matrices in COOrdinate format.

    Parameters
    ----------
    suite2p_stats : np.ndarray
        A numpy array loaded from a Suite2P `stat.npy` output file.
        Each element in the array is a dictionary containing information about
        a unique ROI.

        Each ROI dictionary contains the following fields:
        ['ypix', 'lam', 'xpix', 'mrs', 'mrs0', 'compact', 'med', 'npix',
        'footprint', 'npix_norm', 'overlap', 'ipix', 'radius',
        'aspect_ratio', 'skew', 'std']

    movie_shape : Tuple[int, int]
        The frame shape of the movie from which ROIs were extracted in order
        of: (height, width).

    Returns
    -------
    List[coo_matrix]
        A list of coo matrices. Each matrix represents an ROI.
    """

    coo_rois = [coo_matrix((roi['lam'], (roi['ypix'], roi['xpix'])),
                           shape=movie_shape)
                for roi in suite2p_stats]

    return coo_rois


def binarize_roi_mask(roi_mask: coo_matrix,
                      absolute_threshold: Optional[float] = None,
                      quantile: float = 0.1) -> coo_matrix:
    """Binarize a coo_matrix representing an ROI mask.

    Parameters
    ----------
    roi_mask : coo_matrix
        An ROI mask in coo_matrix format.
    absolute_threshold : Optional[float], optional
        ROI data (Suite2P weights) above the threshold will be
        set to 1 and set to 0 otherwise. If None is provided, the threshold
        will be determined via quantile. By default None.
    quantile : float, optional
        Compute the specified quantile and use it as the absolute_threshold,
        by default 0.1. This parameter will be ignored if an absolute_threshold
        is provided.

    Returns
    -------
    coo_matrix
        A binarized version of the coo_matrix in `np.uint8` format.
    """
    if absolute_threshold is None:
        absolute_threshold = np.quantile(roi_mask.data, quantile)

    binarized_mask = roi_mask.copy()

    over_thresh = (binarized_mask.data > absolute_threshold)
    under_thresh = np.logical_not(over_thresh)

    binarized_mask.data[over_thresh] = 1
    binarized_mask.data[under_thresh] = 0
    binarized_mask.eliminate_zeros()

    return binarized_mask.astype(np.uint8)
