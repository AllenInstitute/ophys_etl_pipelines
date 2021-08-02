from typing import List, Dict, Tuple
import numpy as np
import pathlib
import json
from skimage.measure import label as skimage_label
from scipy.spatial.distance import cdist
from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import get_roi_pixels


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
    new_roi = OphysROI(x0=roi['x'],
                       y0=roi['y'],
                       width=roi['width'],
                       height=roi['height'],
                       mask_matrix=roi['mask'],
                       roi_id=roi['id'],
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
                         valid_roi=roi.valid_roi,
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
    for roi in roi_list:
        if "valid" in roi:
            roi["valid_roi"] = roi.pop("valid")
        if "mask" in roi:
            roi["mask_matrix"] = roi.pop("mask")
        new_list.append(roi)
    return new_list


def roi_list_from_file(file_path: pathlib.Path) -> List[OphysROI]:
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
    output_list = []
    with open(file_path, 'rb') as in_file:
        roi_data_list = json.load(in_file)
        roi_data_list = convert_roi_keys(roi_data_list)
        for roi_data in roi_data_list:
            roi = OphysROI.from_schema_dict(roi_data)
            output_list.append(roi)
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
