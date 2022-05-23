from matplotlib import cm as mplt_cm
import math
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import copy
import networkx
from scipy.sparse import coo_matrix
from scipy.spatial.distance import cdist
from ophys_etl.utils.motion_border import MaxFrameShift
from ophys_etl.types import DenseROI, ExtractROI, OphysROI
from skimage.morphology import binary_opening, binary_closing, disk


def full_mask_constructor(mask_matrix: List[List[bool]], x: int, y: int,
                          shape: Tuple[int, int]) -> np.ndarray:
    """make a full-framed mask of the ROI
    Parameters
    ----------
    mask_matrix: List[List[bool]]
        cropped mask
    x: int
        column offset of mask_matrix
    y:
        row offset of mask_matrix
    shape: Tuple[int, int]
        The frame shape of the movie from which ROIs were extracted in order
        of: (height, width).

    Returns
    -------
    mask: numpy.ndarray
        boolean array, same shape as the input shape

    Notes
    -----
    DenseROI masks are generated by saving only the smallest bounding box that
    contains the ROI as a dense array. This function takes the bounded ROI as
    well as as the x and y top left position of the bounding box and
    regenerates the ROI as it would appear in the original field of view.

    """
    height, width = np.array(mask_matrix).shape
    mask = np.pad(mask_matrix,
                  pad_width=((y, shape[0] - height - y),
                             (x, shape[1] - width - x)),
                  mode='constant')
    return mask


def roi_from_full_mask(roi: DenseROI, mask: np.ndarray
                       ) -> Union[DenseROI, None]:
    """replaces mask and related keys in roi with new mask

    Parameters
    ----------
    roi: DenseROI
        roi for mask replacement
    mask: numpy.ndarray
        boolean, assumed to be full frame

    Returns
    -------
    roi: DenseROI
        roi with mask replaced, or None if mask has no entries

    """
    where = np.where(mask)
    if where[0].size == 0:
        return None
    roi['x'] = int(where[1].min())
    roi['width'] = int(where[1].ptp() + 1)
    roi['y'] = int(where[0].min())
    roi['height'] = int(where[0].ptp() + 1)
    list_mask = []
    for y in range(roi['y'], roi['y'] + roi['height']):
        list_mask.append(mask[y, roi['x']:(roi['x'] + roi['width'])].tolist())
    roi['mask_matrix'] = list_mask
    return roi


def morphological_transform(roi: DenseROI, shape: Tuple[int, int]
                            ) -> Union[DenseROI, None]:
    """performs a closing followed by an opening to clean up pixelated
    appearance of ROIs

    Parameters
    ----------
    roi: DenseROI
        roi to transform
    shape: Tuple[int, int]
        The frame shape of the movie from which ROIs were extracted in order
        of: (height, width).

    Returns
    -------
    roi: DenseROI
        transformed roi or None if empty after transform

    """

    mask = full_mask_constructor(roi['mask_matrix'], roi['x'], roi['y'], shape)
    structuring_element = disk(radius=1)
    mask = binary_closing(mask, selem=structuring_element)
    mask = binary_opening(mask, selem=structuring_element)
    new_roi = roi_from_full_mask(roi, mask)
    return new_roi


def dense_to_extract(roi: DenseROI) -> ExtractROI:
    """reformat from expected output format from binarization
    to expected input format of extract_traces

    Parameters
    ----------
    roi: DenseROI
        an ROI in the output format of binarization

    Returns
    -------
    exroi: ExtractROI
        an ROI in the input format for extract_traces

    """
    exroi = ExtractROI(
            id=roi['id'],
            x=roi['x'],
            y=roi['y'],
            width=roi['width'],
            height=roi['height'],
            valid=roi['valid_roi'],
            mask=roi['mask_matrix'])
    return exroi


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
        ROI data (Suite2P weights) above and equal to the threshold will be
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
    binarized_mask.data = np.where(binarized_mask.data >= absolute_threshold,
                                   1, 0)
    binarized_mask.eliminate_zeros()

    return binarized_mask


def roi_bounds(roi_mask: coo_matrix) -> Union[Tuple[int, int, int, int], None]:
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
        or None if mask is empty
    """

    if roi_mask.row.size == 0 | roi_mask.col.size == 0:
        return None

    min_row = roi_mask.row.min()
    min_col = roi_mask.col.min()
    # Need to add 1 to max indices to get correct slicing upper bound
    max_row = roi_mask.row.max() + 1
    max_col = roi_mask.col.max() + 1

    return (min_row, max_row, min_col, max_col)


def crop_roi_mask(roi_mask: coo_matrix) -> coo_matrix:
    """Crop ROI mask into smallest rectangle that fits all nonzero elements

    Parameters
    ----------
    roi_mask : coo_matrix

    Returns
    -------
    coo_matrix
        A cropped ROI mask or None if coo_matrix is empty

    """

    bounds = roi_bounds(roi_mask)
    if bounds is None:
        return None

    # Convert coo to csr matrix so we can take advantage of indexing
    cropped_mask = roi_mask.tocsr()[bounds[0]:bounds[1], bounds[2]:bounds[3]]

    return cropped_mask.tocoo()


def coo_rois_to_lims_compatible(coo_masks: List[coo_matrix],
                                max_correction_vals: MaxFrameShift,
                                movie_shape: Tuple[int, int],
                                npixel_threshold: int,
                                ) -> List[DenseROI]:
    """
    Converts coo formatted ROIs to lims compatible format.

    Parameters
    ----------
    coo_masks: List[coo_matrix]
        A list of scipy coo_matrices representing ROI masks, each element of
        list is a unique ROI.
    max_correction_vals: MaxFrameShift
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
    List[DenseROI]
        converted rois into LIMS-standard form

    """
    compatible_rois = []
    for temp_id, coo_mask in enumerate(coo_masks):
        compatible_roi = _coo_mask_to_LIMS_compatible_format(coo_mask)
        if compatible_roi is None:
            continue

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


def _coo_mask_to_LIMS_compatible_format(coo_mask: coo_matrix
                                        ) -> Union[DenseROI, None]:
    """
    This functions transforms ROI mask data from COO format
    to the LIMS expected format.
    Parameters
    ----------
    coo_mask: coo_matrix
        The coo roi matrix to be converted

    Returns
    -------
    DenseROI
       or None if the coo_mask is empty

    """
    bounds = roi_bounds(coo_mask)
    if bounds is None:
        return None

    height = bounds[1] - bounds[0]
    width = bounds[3] - bounds[2]
    mask_matrix = crop_roi_mask(coo_mask).toarray()
    mask_matrix = np.array(mask_matrix, dtype=bool)
    compatible_roi = DenseROI(
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


def _motion_exclusion(roi: DenseROI, movie_shape: Tuple[int, int]) -> bool:
    """
    Parameters
    ----------
    roi: DenseROI
        The ROI to check
    movie_shape: Tuple[int, int]
        The frame shape of the movie from which ROIs were extracted in order
        of: (height, width).

    Returns
    -------
    valid: bool
        True, if the ROI in question does not interesect with a motion
        exclusion area. False if it does.

    """
    # A rightward shift increases the min 'valid' left border of the movie
    l_inset = math.ceil(roi['max_correction_right'])
    # Conversely, a leftward shift reduces the 'valid' right border
    r_inset = math.floor(movie_shape[1] - roi['max_correction_left'])
    t_inset = math.ceil(roi['max_correction_down'])
    b_inset = math.floor(movie_shape[0] - roi['max_correction_up'])

    valid = ((roi['x'] >= l_inset)
             & (roi['x'] + roi['width'] <= r_inset)
             & (roi['y'] >= t_inset)
             & (roi['y'] + roi['height'] <= b_inset))

    return valid


def _small_size_exclusion(roi: DenseROI, npixel_threshold: int) -> bool:
    """
    Parameters
    ----------
    roi: DenseROI
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


def _check_exclusion(compatible_roi: DenseROI,
                     movie_shape: Tuple[int, int],
                     npixel_threshold: int) -> List[str]:
    """
    Check ROI for different possible exclusions

    Parameters
    ----------
    compatible_roi: DenseROI
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


def extract_roi_to_ophys_roi(roi: ExtractROI) -> OphysROI:
    """
    Convert an ExtractROI to an equivalent OphysROI

    Parameters
    ----------
    ExtractROI

    Returns
    -------
    OphysROI
    """
    new_roi = OphysROI(x0=int(roi['x']),
                       y0=int(roi['y']),
                       width=int(roi['width']),
                       height=int(roi['height']),
                       mask_matrix=roi['mask'],
                       roi_id=int(roi['id']),
                       valid_roi=roi['valid'])

    return new_roi


def ophys_roi_to_extract_roi(roi: OphysROI) -> ExtractROI:
    """
    Convert at OphysROI to an equivalent ExtractROI

    Parameters
    ----------
    OphysROI

    Returns
    -------
    ExtractROI
    """
    mask = []
    for roi_row in roi.mask_matrix:
        row = []
        for el in roi_row:
            if el:
                row.append(True)
            else:
                row.append(False)
        mask.append(row)

    new_roi = ExtractROI(x=roi.x0,
                         y=roi.y0,
                         width=roi.width,
                         height=roi.height,
                         mask=mask,
                         valid=roi.valid_roi,
                         id=roi.roi_id)
    return new_roi


def sanitize_extract_roi_list(
        input_roi_list: List[Dict]) -> List[ExtractROI]:
    """
    There are, unfortunately, two ROI serialization schemes floating
    around in our code base. This method converts the one that is
    incompatible with ExtractROI to a list of ExtractROI. Specifically,
    it converts

    valid_roi -> valid
    mask_matrix -> mask
    roi_id - > id

    Parameters
    ----------
    input_roi_list: List[Dict]
        List of ROIs represented as dicts which ar inconsistent
        with ExtractROI

    Returns
    -------
    output_roi_list: List[ExtractROI]
    """
    output_roi_list = []
    for roi in input_roi_list:
        new_roi = copy.deepcopy(roi)
        if 'valid_roi' in new_roi:
            new_roi['valid'] = new_roi.pop('valid_roi')
        if 'mask_matrix' in new_roi:
            new_roi['mask'] = new_roi.pop('mask_matrix')
        if 'roi_id' in new_roi:
            new_roi['id'] = new_roi.pop('roi_id')
        output_roi_list.append(new_roi)
    return output_roi_list


def _do_rois_abut(array_0: np.ndarray,
                  array_1: np.ndarray,
                  pixel_distance: float = np.sqrt(2)) -> bool:
    """
    Function that does the work behind user-facing do_rois_abut.

    This function takes in two arrays of pixel coordinates
    calculates the distance between every pair of pixels
    across the two arrays. If the minimum distance is less
    than or equal pixel_distance, it returns True. If not,
    it returns False.

    Parameters
    ----------
    array_0: np.ndarray
        Array of the first set of pixels. Shape is (npix0, 2).
        array_0[:, 0] are the row coodinates of the pixels
        in array_0. array_0[:, 1] are the column coordinates.

    array_1: np.ndarray
        Same as array_0 for the second array of pixels

    pixel_distance: float
        Maximum distance two arrays can be from each other
        at their closest point and still be considered
        to abut (default: sqrt(2)).

    Return
    ------
    boolean
    """
    distances = cdist(array_0, array_1, metric='euclidean')
    if distances.min() <= pixel_distance:
        return True
    return False


def do_rois_abut(roi0: OphysROI,
                 roi1: OphysROI,
                 pixel_distance: float = np.sqrt(2)) -> bool:
    """
    Returns True if ROIs are within pixel_distance of each other at any point.

    Parameters
    ----------
    roi0: OphysROI

    roi1: OphysROI

    pixel_distance: float
        The maximum distance from each other the ROIs can be at
        their closest point and still be considered to abut.
        (Default: np.sqrt(2))

    Returns
    -------
    boolean

    Notes
    -----
    pixel_distance is such that if two boundaries are next to each other,
    that corresponds to pixel_distance=1; pixel_distance=2 corresponds
    to 1 blank pixel between ROIs
    """

    return _do_rois_abut(roi0.global_pixel_array,
                         roi1.global_pixel_array,
                         pixel_distance=pixel_distance)


def get_roi_color_map(
        roi_list: Union[List[OphysROI],
                        List[Dict]]) -> Dict[int, Tuple[int, int, int]]:
    """
    Take a list of OphysROI and return a dict mapping ROI ID
    to RGB color so that no ROIs that touch have the same color

    Parametrs
    ---------
    roi_list: Union[List[OphysROI], List[Dict]]

    Returns
    -------
    color_map: Dict[int, Tuple[int, int, int]]
    """
    if not isinstance(roi_list[0], OphysROI):
        roi_list = [extract_roi_to_ophys_roi(roi)
                    for roi in sanitize_extract_roi_list(roi_list)]

    roi_graph = networkx.Graph()
    for roi in roi_list:
        roi_graph.add_node(roi.roi_id)
    for ii in range(len(roi_list)):
        roi0 = roi_list[ii]
        for jj in range(ii+1, len(roi_list)):
            roi1 = roi_list[jj]

            # value of 5 is so that singleton ROIs that
            # are near each other do not get assigned
            # the same color
            abut = do_rois_abut(roi0, roi1, 5.0)
            if abut:
                roi_graph.add_edge(roi0.roi_id, roi1.roi_id)
                roi_graph.add_edge(roi1.roi_id, roi0.roi_id)

    nx_coloring = networkx.greedy_color(roi_graph)
    n_colors = len(set(nx_coloring.values()))

    mplt_color_map = mplt_cm.jet

    # create a list of colors based on the matplotlib color map
    raw_color_list = []
    for ii in range(n_colors):
        color = mplt_color_map(0.8*(1.0+ii)/(n_colors+1.0))
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        raw_color_list.append(color)

    # re-order colors so that colors that are adjacent in index
    # have higher contrast
    step = max(n_colors//3, 1)
    color_list = []
    for i0 in range(step):
        for ii in range(i0, n_colors, step):
            this_color = raw_color_list[ii]
            color_list.append(this_color)

    # reverse color list, since matplotlib.cm.jet will
    # assign a dark blue as color_list[0], which isn't
    # great for contrast
    color_list.reverse()

    color_map = {}
    for roi_id in nx_coloring:
        color_map[roi_id] = color_list[nx_coloring[roi_id]]
    return color_map


def get_roi_list_in_fov(
        roi_list: List[ExtractROI],
        origin: Tuple[int, int],
        frame_shape: Tuple[int, int]) -> List[ExtractROI]:
    """
    Select only the ROIs whose thumbnails intersect
    with a specified field of view

    Parameters
    ----------
    roi_list: List[ExtractROI]

    origin: Tuple[int, int]
        The origin in row, col coordinates of the field of view

    frame_shape: Tuple[int, int]
        (height, width) of the field of view

    Returns
    -------
    List[ExtractROI]
    """

    global_r0 = origin[0]
    global_r1 = global_r0 + frame_shape[0]
    global_c0 = origin[1]
    global_c1 = global_c0 + frame_shape[1]

    output = []
    for roi in roi_list:
        r0 = roi['y']
        r1 = r0+roi['height']
        c0 = roi['x']
        c1 = c0+roi['width']
        if r1 < global_r0:
            continue
        elif r0 >= global_r1:
            continue
        elif c1 < global_c0:
            continue
        elif c0 >= global_c1:
            continue

        output.append(roi)
    return output


def clip_roi(
        roi: ExtractROI,
        full_fov_shape: Tuple[int, int],
        row_bounds: Tuple[int, int],
        col_bounds: Tuple[int, int]) -> ExtractROI:
    """
    Retrun an ExtractROI that encodes the same
    shape in the full FOV window specified by
    row_bounds and col_bounds.

    Depending on the bounds specified, this may
    clip pixels from the ROI.
    """
    ophys_roi = extract_roi_to_ophys_roi(roi)
    mask = np.zeros(full_fov_shape, dtype=bool)
    pixel_array = ophys_roi.global_pixel_array.transpose()
    mask[pixel_array[0], pixel_array[1]] = True
    mask = mask[row_bounds[0]:row_bounds[1],
                col_bounds[0]:col_bounds[1]]
    new_roi = OphysROI(
                roi_id=ophys_roi.roi_id,
                x0=0,
                y0=0,
                width=int(mask.shape[1]),
                height=int(mask.shape[0]),
                valid_roi=True,
                mask_matrix=mask)
    return ophys_roi_to_extract_roi(new_roi)
