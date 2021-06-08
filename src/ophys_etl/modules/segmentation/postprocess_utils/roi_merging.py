import numpy as np
from scipy.spatial.distance import cdist
from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import get_roi_pixels


def extract_roi_to_ophys_roi(roi):
    new_roi = OphysROI(x0=roi['x'],
                       y0=roi['y'],
                       width=roi['width'],
                       height=roi['height'],
                       mask_matrix=roi['mask_matrix'],
                       roi_id=roi['id'],
                       valid_roi=True)

    return new_roi


def ophys_roi_to_extract_roi(roi):
    new_roi = ExtractROI(x=roi.x0,
                         y=roi.y0,
                         width=roi.width,
                         height=roi.height,
                         mask=[list(i) for i in roi.mask_matrix],
                         valid_roi=True,
                         id=roi.roi_id)
    return new_roi


def merge_rois(roi0: OphysROI,
               roi1: OphysROI,
               new_roi_id: int) -> OphysROI:

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


def _get_bdry_array(roi: OphysROI):
    """
    get Nx2 array of pixels (in global coordinates)
    that are on boundary of ROI
    """
    bdry_mask = roi.boundary_mask
    n_bdry = bdry_mask.sum()
    bdry_array = -1*np.ones((n_bdry, 2), dtype=int)
    i_pix = 0
    for ir in range(roi.height):
        row = ir+roi.y0
        for ic in range(roi.width):
            col =ic+roi.x0
            if not bdry_mask[ir, ic]:
                continue

            bdry_array[i_pix, 0] = row
            bdry_array[i_pix, 1] = col
            i_pix += 1

    if bdry_array.min() < 0:
        raise RuntimeError("did not assign all boundary pixels")

    return bdry_array


def do_rois_abut(roi0: OphysROI,
                 roi1: OphysROI,
                 dpix: float = 1) -> bool:
    """
    Returns True if ROIs are within dpix of each other at any point along
    their boundaries

    Note: dpix is such that if two boundaries are next to each other,
    that is dpix=1; dpix=2 is a 1 blank pixel between ROIs
    """

    # do the ROIs overlap
    pixel_dict = get_roi_pixels([roi0, roi1])
    intersection = pixel_dict[roi0.roi_id].intersection(pixel_dict[roi1.roi_id])
    if len(intersection) > 0:
        return True

    bdry_0 = _get_bdry_array(roi0)
    bdry_1 = _get_bdry_array(roi1)

    distances = cdist(bdry_0, bdry_1, metric='euclidean')
    if distances.min() <= dpix:
        return True
    return False
