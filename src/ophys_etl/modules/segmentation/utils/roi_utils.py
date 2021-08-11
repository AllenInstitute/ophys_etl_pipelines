from typing import List, Dict, Tuple
import numpy as np
import pathlib
import json
import copy
from skimage.measure import label as skimage_label
from scipy.spatial.distance import cdist
from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import get_roi_pixels


def check_matching_extract_roi_lists(listA: List[ExtractROI],
                                     listB: List[ExtractROI]) -> None:
    """check that 2 lists of ROIs are the same, order independent

    Parameters
    ----------
    listA: List[ExtractROI]
        first list of ROIs
    listB: List[ExtractROI]
        second list of ROIs

    Raises
    ------
    AssertionError if the lists do not match

    """
    # list of IDs match
    idsA = {i["id"] for i in listA}
    idsB = {i["id"] for i in listB}
    assert idsA == idsB, ("ids in ROI lists do not match. "
                          f"{idsA - idsB} in first list but not second list. "
                          f"{idsB - idsA} in second list but not first list. ")

    # ROIs match
    lookupA = {i["id"]: i for i in listA}
    lookupB = {i["id"]: i for i in listB}
    for idkey, roi in lookupA.items():
        assert roi == lookupB[idkey], (f"roi with ID {idkey} does not match "
                                       "between the 2 lists.")


def serialize_extract_roi_list(rois: List[ExtractROI]) -> bytes:
    """converts a list of ROIs to bytes that can be stored
    in an hdf5 dataset.

    Parameters
    ----------
    rois: List[ExtractROI]
        the list of ROI dictionaries

    Returns
    -------
    serialized: bytes
        the serialized representation

    """
    serialized = json.dumps(rois).encode("utf-8")
    return serialized


def deserialize_extract_roi_list(serialized: bytes) -> List[ExtractROI]:
    """deserializes bytest into a list of ROIs

    Parameters
    ----------
    serialized: bytes
        the serialized representation

    Returns
    -------
    rois: List[ExtractROI]
        the list of ROI dictionaries

    """
    rois = json.loads(serialized.decode("utf-8"))
    return rois


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


def convert_to_lims_roi(origin: Tuple[int, int],
                        mask: np.ndarray,
                        roi_id: int = 0) -> ExtractROI:
    """
    Convert an origin and a pixel mask into a LIMS-friendly
    JSONized ROI

    Parameters
    ----------
    origin: Tuple[int, int]
        The global coordinates of the upper right corner of the pixel mask

    mask: np.ndarray
        A 2D array of booleans marked as True at the ROI's pixels

    roi_id: int
        default: 0

    Returns
    --------
    roi: dict
        an ExtractROI matching the input data
    """
    # trim mask
    valid = np.argwhere(mask)
    row0 = valid[:, 0].min()
    row1 = valid[:, 0].max() + 1
    col0 = valid[:, 1].min()
    col1 = valid[:, 1].max() + 1

    new_mask = mask[row0:row1, col0:col1]
    roi = ExtractROI(id=roi_id,
                     x=int(origin[1]+col0),
                     y=int(origin[0]+row0),
                     width=int(col1-col0),
                     height=int(row1-row0),
                     valid=True,
                     mask=[i.tolist() for i in new_mask])
    return roi


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


def merge_rois(roi0: OphysROI,
               roi1: OphysROI,
               new_roi_id: int) -> OphysROI:
    """
    Merge two OphysROIs into one OphysROI whose
    mask is the union of the two input masks

    Parameters
    ----------
    roi0: OphysROI

    roi1: OphysROI

    new_roi_id: int
        The roi_id to assign to the output ROI

    Returns
    -------
    OphysROI
    """

    xmin0 = roi0.x0
    xmax0 = roi0.x0+roi0.width
    ymin0 = roi0.y0
    ymax0 = roi0.y0+roi0.height
    xmin1 = roi1.x0
    xmax1 = roi1.x0+roi1.width
    ymin1 = roi1.y0
    ymax1 = roi1.y0+roi1.height

    xmin = min(xmin0, xmin1)
    xmax = max(xmax0, xmax1)
    ymin = min(ymin0, ymin1)
    ymax = max(ymax0, ymax1)

    width = xmax-xmin
    height = ymax-ymin

    mask = np.zeros((height, width), dtype=bool)

    pixel_dict = get_roi_pixels([roi0, roi1])
    for roi_id in pixel_dict:
        roi_mask = pixel_dict[roi_id]
        for pixel in roi_mask:
            mask[pixel[1]-ymin, pixel[0]-xmin] = True

    new_roi = OphysROI(x0=xmin,
                       y0=ymin,
                       width=width,
                       height=height,
                       mask_matrix=mask,
                       roi_id=new_roi_id,
                       valid_roi=True)

    return new_roi


def sub_video_from_roi(roi: OphysROI,
                       video_data: np.ndarray) -> np.ndarray:
    """
    Get a sub-video that is flattened in space corresponding
    to the video data at the ROI

    Parameters
    ----------
    roi: OphysROI

    video_data: np.ndarray
        Shape is (ntime, nrows, ncols)

    Returns
    -------
    sub_video: np.ndarray
        Shape is (ntime, npix) where npix is the number
        of pixels marked True in the ROI
    """

    xmin = roi.x0
    ymin = roi.y0
    xmax = roi.x0+roi.width
    ymax = roi.y0+roi.height

    sub_video = video_data[:, ymin:ymax, xmin:xmax]

    mask = roi.mask_matrix
    sub_video = sub_video[:, mask].reshape(video_data.shape[0], -1)
    return sub_video


def intersection_over_union(roi0: OphysROI,
                            roi1: OphysROI) -> float:
    """
    Return the intersection over union of two ROIs relative
    to each other

    Parameters
    ----------
    roi0: OphysROI

    roi1: OphysROI

    Returns
    -------
    iou: float
        """
    pix0 = roi0.global_pixel_set
    pix1 = roi1.global_pixel_set
    ii = len(pix0.intersection(pix1))
    uu = len(pix0.union(pix1))
    return float(ii)/float(uu)


def convert_roi_keys(roi_list: List[Dict]) -> List[Dict]:
    """convert from key names expected by ExtractROI to
    key names expected by OphysROI
    """
    new_list = []
    for old_roi in roi_list:
        roi = copy.deepcopy(old_roi)
        if "valid" in roi:
            roi["valid_roi"] = roi.pop("valid")
        if "mask" in roi:
            roi["mask_matrix"] = roi.pop("mask")
        new_list.append(roi)
    return new_list


def ophys_roi_list_from_deserialized(
        rois: List[ExtractROI]) -> List[OphysROI]:
    """Convert a deserialized ExtractROI list to a list
    of OphysROI objects.

    Parameters
    ----------
    rois: List[ExtractROI]
        list of ExtractROI

    Returns
    -------
    output_list: List[OphysROI]
        converted list of OphysROI

    """
    roi_data_list = convert_roi_keys(rois)
    output_list = []
    for roi_data in roi_data_list:
        roi = OphysROI.from_schema_dict(roi_data)
        output_list.append(roi)
    return output_list


def ophys_roi_list_from_file(file_path: pathlib.Path) -> List[OphysROI]:
    """
    Read in a JSONized file of ExtractROIs; return a list of
    OphysROIs

    Parameters
    ----------
    file_path: pathlib.Path

    Returns
    -------
    List[OphysROI]

    """
    with open(file_path, 'rb') as in_file:
        roi_data_list = json.load(in_file)
    output_list = ophys_roi_list_from_deserialized(roi_data_list)
    return output_list


def select_contiguous_region(
        seed_pt: Tuple[int, int],
        input_mask: np.ndarray) -> np.ndarray:
    """
    Select only the contiguous region of an ROI mask that contains
    a specified seed_pt

    Parameters
    ----------
    seed_pt: Tuple[int]
        The (row, col) coordinate of the seed point that
        must be contained in the returned mask

    input_mask: np.ndarray
        A mask of booleans

    Returns
    -------
    contiguous_mask: np.ndarray
        A mask of booleans corresponding to the contiguous
        block of True pixels in input_mask that contains seed_pt
    """
    if seed_pt[0] >= input_mask.shape[0] or seed_pt[1] >= input_mask.shape[1]:
        msg = f"seed_pt: {seed_pt}\n"
        msg += f"does not exist in mask with shape {input_mask.shape}"
        raise IndexError(msg)

    if not input_mask[seed_pt[0], seed_pt[1]]:
        return np.zeros(input_mask.shape, dtype=bool)

    labeled_img = skimage_label(input_mask, connectivity=2)
    seed_label = labeled_img[seed_pt[0], seed_pt[1]]
    return (labeled_img == seed_label)


def background_mask_from_roi_list(
        roi_list: List[OphysROI],
        img_shape: Tuple[int]) -> np.ndarray:
    """
    Take a list of ROIs. Return an np.ndarray of booleans marked
    as False on any pixel that is included in the ROIs.

    Parameters
    ----------
    roi_list: List[OphysROI]

    img_shape: Tuple[int]
        The shape of the output mask array

    Returns
    -------
    background_mask: np.ndarray
    """

    background_mask = np.ones(img_shape, dtype=bool)
    for roi in roi_list:
        rows = roi.global_pixel_array[:, 0]
        cols = roi.global_pixel_array[:, 1]
        background_mask[rows, cols] = False
    return background_mask


def select_window_from_background(
        roi: OphysROI,
        background_mask: np.ndarray,
        n_desired_background: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Get the (rowmin, rowmax), (colmin, colmax) bounds of a window
    centered on an ROI that contains at least a specified number
    of background pixels

    Parameters
    ----------
    roi: OphysROI

    background_mask: np.ndarray
        A mask covering the full field of view that is marked True
        for every pixel that is a background pixel.

    n_desired_background: int
        Minimum number of background pixels to be returned in the
        window

    Returns
    -------
    bounds: Tuple[Tuple[int, int], Tuple[int, int]]
        Of the form ((rowmin, rowmax), (colmin, colmax))

    Notes
    -----
    If it is not possible to find a window with n_desired_background
    pixels, bounds will just encompass the full field of view
    """
    if n_desired_background >= background_mask.sum():
        return ((0, background_mask.shape[0]),
                (0, background_mask.shape[1]))

    centroid_row = np.round(roi.centroid_y).astype(int)
    centroid_col = np.round(roi.centroid_x).astype(int)

    area_estimate = roi.area + n_desired_background
    pixel_radius = max(np.round(np.sqrt(area_estimate)).astype(int)//2,
                       1)
    n_background = 0
    while n_background < n_desired_background:
        rowmin = max(0, centroid_row-pixel_radius)
        rowmax = min(background_mask.shape[0],
                     centroid_row+pixel_radius+1)
        colmin = max(0, centroid_col-pixel_radius)
        colmax = min(background_mask.shape[1],
                     centroid_col+pixel_radius+1)
        local_mask = background_mask[rowmin:rowmax, colmin:colmax]
        n_background = local_mask.sum()
        pixel_radius += 1
        if local_mask.sum() == background_mask.sum():
            # we are now using all of the background pixels
            break

    return ((rowmin, rowmax), (colmin, colmax))


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
