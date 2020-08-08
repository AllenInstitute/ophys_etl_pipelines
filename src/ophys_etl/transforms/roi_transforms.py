from typing import List, Optional, Tuple
import numpy as np
from scipy.sparse import coo_matrix
import sys
from ophys_etl.extractors.motion_correction import MotionBorder

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class StandardROI(TypedDict):
    id: int
    x: int
    y: int
    width: int
    height: int
    valid_roi: bool
    mask_matrix: List[List[bool]]
    max_correction_up: int
    max_correction_down: int
    max_correction_left: int
    max_correction_right: int
    mask_image_plane: int
    exclusion_labels: List[str]


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


def coo_rois_to_lims_compatible(coo_masks: List[coo_matrix],
                                max_correction_vals: MotionBorder,
                                movie_shape: Tuple[int, int],
                                npixel_threshold: int,
                                ) -> List[StandardROI]:
    """
    Converts coo formatted ROIs to lims compatible format.

    Parameters
    ----------
    coo_masks: List[coo_matrix]
        A list of scipy coo_matrices representing ROI masks, each element of
        list is a unique ROI.
    max_correction_vals: MotionBorder
        Named tuple containing the max motion correction values identified
        in the motion correction step of ophys segmentation pipeline.
        Name tuple has the following names: ['left', 'right', 'up', 'down'].
    movie_shape: Tuple[int, int]
        The frame shape of the movie from which ROIs were extracted in order
        of: (height, width).
    npixel_threshold: int
        ROIs with fewer pixels than this will be labeled as invalid and small
        size

    Returns
    -------
    List[StandardROI]
        converted rois into LIMS-standard form

    """
    compatible_rois = []
    for temp_id, coo_mask in enumerate(coo_masks):
        compatible_roi = _coo_mask_to_LIMS_compatible_format(coo_mask)
        compatible_roi['id'] = temp_id  # popped off when writing to LIMs
        compatible_roi['max_correction_up'] = max_correction_vals.up
        compatible_roi['max_correction_down'] = max_correction_vals.down
        compatible_roi['max_correction_right'] = max_correction_vals.right
        compatible_roi['max_correction_left'] = max_correction_vals.left

        labels = _check_exclusion(compatible_roi,
                                  movie_shape,
                                  npixel_threshold)
        compatible_roi['exclusion_labels'] = labels
        compatible_roi['valid_roi'] = not any(labels)

        compatible_rois.append(compatible_roi)

    return compatible_rois


def _coo_mask_to_LIMS_compatible_format(coo_mask: coo_matrix) -> StandardROI:
    """
    This functions transforms ROI mask data from COO format
    to the LIMS expected format.
    Parameters
    ----------
    coo_mask: coo_matrix
        The coo roi matrix to be converted

    Returns
    -------
    StandardROI

    """
    bounds = roi_bounds(coo_mask)
    height = bounds[1] - bounds[0]
    width = bounds[3] - bounds[2]
    mask_matrix = crop_roi_mask(coo_mask).toarray()
    mask_matrix = np.array(mask_matrix, dtype=bool)
    compatible_roi = StandardROI(
        x=int(bounds[2]),
        y=int(bounds[0]),
        width=int(width),
        height=int(height),
        mask_matrix=mask_matrix.tolist(),
        # following are placeholders
        valid_roi=True,
        mask_image_plane=0,
        exclusion_labels=[],
        id=-1,
        max_correction_up=-1,
        max_correction_down=-1,
        max_correction_left=-1,
        max_correction_right=-1)
    return compatible_roi


def _motion_exclusion(roi: StandardROI, movie_shape: Tuple[int, int]) -> bool:
    """
    Parameters
    ----------
    roi: StandardROI
        the ROI to check
    movie_shape: Tuple[int, int]
        The frame shape of the movie from which ROIs were extracted in order
        of: (height, width).

    Returns
    -------
    valid: bool
        whether this ROI is valid on motion exclusion

    """
    furthest_right_pixel = roi['x'] + roi['width'] - 1
    furthest_down_pixel = roi['y'] + roi['height'] - 1
    right_limit = movie_shape[1] - roi['max_correction_right']
    bottom_limit = movie_shape[0] - roi['max_correction_down']

    valid = ((roi['x'] > roi['max_correction_left'] - 1) &
             (roi['y'] > roi['max_correction_up'] - 1) &
             (furthest_right_pixel < right_limit) &
             (furthest_down_pixel < bottom_limit))

    return valid


def _small_size_exclusion(roi: StandardROI, npixel_threshold: int) -> bool:
    """
    Parameters
    ----------
    roi: StandardROI
        the ROI to check
    npixel_threshold: int
        ROIs with fewer pixels than this will be labeled as invalid and small
        size

    Returns
    -------
    valid: bool
        whether this ROI is valid on small size exclusion

    """
    npixels = sum([sum(i) for i in roi['mask_matrix']])
    valid = npixels > npixel_threshold
    return valid


def _check_exclusion(compatible_roi: StandardROI,
                     movie_shape: Tuple[int, int],
                     npixel_threshold: int) -> List[str]:
    """
    Check ROI for different possible exclusions

    Parameters
    ----------
    compatible_roi: StandardROI
        the ROI to check
    movie_shape: Tuple[int, int]
        The frame shape of the movie from which ROIs were extracted in order
        of: (height, width).
    npixel_threshold: int
        ROIs with fewer pixels than this will be labeled as invalid and small
        size

    Returns
    -------
    List[str]
        list of exclusion codes, can be empty list

    """
    exclusion_labels = []

    if not _motion_exclusion(compatible_roi, movie_shape):
        exclusion_labels.append('motion_border')

    if not _small_size_exclusion(compatible_roi, npixel_threshold):
        exclusion_labels.append('small_size')

    return exclusion_labels
