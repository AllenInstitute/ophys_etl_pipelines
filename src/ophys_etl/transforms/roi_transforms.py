from typing import List, Optional, Tuple, Dict, Any

import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.transforms.data_loaders import motion_border


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
        A binarized version of the coo_matrix.
    """
    if absolute_threshold is None:
        absolute_threshold = np.quantile(roi_mask.data, quantile)

    binarized_mask = roi_mask.copy()
    binarized_mask.data = np.where(binarized_mask.data > absolute_threshold,
                                   1, 0)
    binarized_mask.eliminate_zeros()

    return binarized_mask


def roi_bounds(roi_mask: coo_matrix) -> Tuple[int, int, int, int]:
    """Get slicing bounds that define the smallest rectangle that contains
    all nonzero ROI elements.

    Note: An empty roi_mask will return all zero bounds.

    Parameters
    ----------
    roi_mask : coo_matrix
        The ROI for which minimal slicing bounds should be determined.

    Returns
    -------
    Tuple[int, int, int, int]
        Slicing bounds to extract an ROI in the following order:
        (min_row, max_row, min_col, max_col)
    """

    if roi_mask.row.size > 0 and roi_mask.col.size > 0:
        min_row = roi_mask.row.min()
        min_col = roi_mask.col.min()
        # Need to add 1 to max indices to get correct slicing upper bound
        max_row = roi_mask.row.max() + 1
        max_col = roi_mask.col.max() + 1
        return (min_row, max_row, min_col, max_col)
    else:
        return (0, 0, 0, 0)


def crop_roi_mask(roi_mask: coo_matrix) -> coo_matrix:
    """Crop ROI mask into smallest rectangle that fits all nonzero elements

    Parameters
    ----------
    roi_mask : coo_matrix

    Returns
    -------
    coo_matrix
        A cropped ROI mask

    Raises
    ------
    ValueError
        Raised if an empty ROI mask is provided
    """
    if roi_mask.row.size > 0 and roi_mask.col.size > 0:
        min_row, max_row, min_col, max_col = roi_bounds(roi_mask)
    else:
        raise ValueError("Cannot crop an empty ROI mask (or mask where all "
                         "elements are zero)")

    # Convert coo to csr matrix so we can take advantage of indexing
    cropped_mask = roi_mask.tocsr()[min_row:max_row, min_col:max_col]

    return cropped_mask.tocoo()


def coo_rois_to_old(coo_masks: List[coo_matrix],
                    max_correction_vals: motion_border,
                    movie_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
    """
    Converts coo formatted ROIs to old format from previous Ophys Segmentation
    implementation. This is a required transformation as the next workflow
    step, Extract Traces, expects ROIs in this format and to reduce work load
    this will not be changed.
    Parameters
    ----------
    coo_masks: List[coo_matrix]
        A list of scipy coo_matrices representing ROI masks, each element of
        list is a unique ROI.
    max_correction_vals: motion_border
        The max motion correction values identified in the motion correction
        step of ophys segmentation pipeline
    movie_shape: Tuple[int, int]
        The frame shape of the movie from which ROIs were extracted in order
        of: (height, width).

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries representing the ROIs in old segmentation
        format, the data contained inside each one is as follows:
        {
            id: int
            x: int
            y: int
            width: int
            height: int
            valid_roi: Bool
            mask_matrix: List[List[Bool]]
            max_correction_up: int
            max_correction_down: int
            max_correction_left: int
            max_correction_right: int
            mask_image_plane: int
            exclusion_labels: List[int] codes are defined in 2019 Ophys Docs
        }
        For details about specific values see design document for 2020 Ophys
        Segmentation Refactor and Update
        """
    old_rois = []
    for temp_id, coo_mask in enumerate(coo_masks):
        old_roi = _coo_mask_to_old_format(coo_mask)
        old_roi['id'] = temp_id  # popped off writing to LIMs
        old_roi['cell_specimen_id'] = temp_id  # updated post nway cellmatching
        old_roi['valid_roi'] = True
        old_roi['max_correction_up'] = max_correction_vals.up
        old_roi['max_correction_down'] = max_correction_vals.down
        old_roi['max_correction_right'] = max_correction_vals.right
        old_roi['max_correction_left'] = max_correction_vals.left
        old_roi['mask_image_plane'] = 0
        old_roi['exclusion_labels'] = []
        _check_motion_exclusion(old_roi, movie_shape)
        old_rois.append(old_roi)
    return old_rois


def _coo_mask_to_old_format(coo_mask: coo_matrix) -> Dict:
    """
    This functions transforms ROI mask data from COO format
    to the old format used in older segmentation. This function
    writes only the data associated with the mask.
    Parameters
    ----------
    coo_mask: coo_matrix
        The coo roi matrix to be converted to old segmentation mask format

    Returns
    -------
    Dict:
        A dictionary that contains the mask data from the coo matrix in the
        old segmentation format
        {
            'x': int (x location of upper left corner of roi in pixels)
            'y': int (y location of upper left corner of roi in pixels)
            'width': int (width of the roi mask in pixels)
            'height': int (height of the roi mask in pixels)
            'mask_matrix': List[List[bool]] (dense matrix of roi mask)
        }
    """
    bounds = roi_bounds(coo_mask)
    height = bounds[1] - bounds[0]
    width = bounds[3] - bounds[2]
    mask_matrix = crop_roi_mask(coo_mask).toarray()
    mask_matrix = np.array(mask_matrix, dtype=bool)
    old_roi = {
        'x': bounds[0],
        'y': bounds[2],
        'width': width,
        'height': height,
        'mask_matrix': mask_matrix.tolist()
    }
    return old_roi


def _check_motion_exclusion(old_roi: Dict,
                            movie_shape: Tuple[int, int]):
    """
    Checks if roi in old styling needs to be excluded as it exists partly
    or wholey outside the bounds of the motion correction border. Assigns
    a string to the list of exclusion labels indicating outside of motion
    border and labels ROI as invalid.
    Parameters
    ----------
    old_roi: Dict
        An ROI stored in old dictionary format that minimally contains the
        following values.
        {
            'x': int (x location of upper left corner of roi in pixels)
            'y': int (y location of upper left corner of roi in pixels)
            'width': int (width of the roi mask in pixels)
            'height': int (height of the roi mask in pixels)
            'max_correction_up': int
            'max_correction_down': int
            'max_correction_left': int
            'max_correction_right': int
        }

    movie_shape: Tuple[int, int]
        The frame shape of the movie from which ROIs were extracted in order
        of: (height, width).
    Returns
    -------

    """
    movie_height, movie_width = movie_shape[0], movie_shape[1]
    furthest_right_pixel = old_roi['x'] + old_roi['width']
    furthest_down_pixel = old_roi['y'] + old_roi['height']
    if (old_roi['x'] <= old_roi['max_correction_left'] or
       old_roi['y'] <= old_roi['max_correction_up'] or
       furthest_right_pixel >= movie_width - old_roi['max_correction_right'] or
       furthest_down_pixel >= movie_height - old_roi['max_correction_down']):
        old_roi['exclusion_labels'].append(7)  # code 7 = motion border error
        old_roi['valid_roi'] = False
