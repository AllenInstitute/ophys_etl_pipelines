import numpy as np
from typing import List, Tuple
import pathlib
import h5py
import time
import multiprocessing
import multiprocessing.managers

from scipy.spatial.distance import cdist
from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import get_roi_pixels

import logging

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def extract_roi_to_ophys_roi(roi):
    new_roi = OphysROI(x0=roi['x'],
                       y0=roi['y'],
                       width=roi['width'],
                       height=roi['height'],
                       mask_matrix=roi['mask'],
                       roi_id=roi['id'],
                       valid_roi=True)

    return new_roi


def ophys_roi_to_extract_roi(roi):
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


def _get_pixel_array(roi: OphysROI):
    """
    get Nx2 array of pixels (in global coordinates)
    that are in the ROI
    """
    mask = roi.mask_matrix
    n_bdry = mask.sum()
    roi_array = -1*np.ones((n_bdry, 2), dtype=int)
    i_pix = 0
    for ir in range(roi.height):
        row = ir+roi.y0
        for ic in range(roi.width):
            col =ic+roi.x0
            if not mask[ir, ic]:
                continue

            roi_array[i_pix, 0] = row
            roi_array[i_pix, 1] = col
            i_pix += 1

    if roi_array.min() < 0:
        raise RuntimeError("did not assign all boundary pixels")

    return roi_array

def _do_rois_abut(array_0: np.ndarray,
                  array_1: np.ndarray,
                  dpix: float = 1) -> bool:

    distances = cdist(array_0, array_1, metric='euclidean')
    if distances.min() <= dpix:
        return True
    return False


def do_rois_abut(roi0: OphysROI,
                 roi1: OphysROI,
                 dpix: float = 1) -> bool:
    """
    Returns True if ROIs are within dpix of each other at any point along
    their boundaries

    Note: dpix is such that if two boundaries are next to each other,
    that is dpix=1; dpix=2 is a 1 blank pixel between ROIs
    """
    array_0 = _get_pixel_array(roi0)
    array_1 = _get_pixel_array(roi1)

    return _do_rois_abut(array_0,
                         array_1,
                         dpix=dpix)


def correlate_sub_videos(sub_video_0: np.ndarray,
                         sub_video_1: np.ndarray,
                         filter_fraction:float) -> np.ndarray:
    """
    correlate pixels in sub_video_0 with pixels in sub_video_1
    """

    assert sub_video_0.shape[0] == sub_video_1.shape[0]
    npix0 = sub_video_0.shape[1]
    npix1 = sub_video_1.shape[1]
    discard = max(0.0, 1.0-filter_fraction)

    corr = np.zeros((npix0, npix1), dtype=float)

    for i_pixel in range(npix0):
        trace0 = sub_video_0[:, i_pixel]
        th = np.quantile(trace0, discard)
        mask = (trace0 > th)
        trace0 = trace0[mask]
        other_video = sub_video_1[mask, :]
        other_mu = np.mean(other_video, axis=0)
        assert other_mu.shape == (npix1, )
        mu = np.mean(trace0)
        var = np.mean((trace0-mu)**2)
        other_video = other_video-other_mu
        other_var = np.mean(other_video**2, axis=0)
        assert other_var.shape == (npix1, )
        numerator = np.dot(other_video.T, (trace0-mu))/mask.sum()
        assert numerator.shape == (npix1, )
        corr[i_pixel,:] = numerator/np.sqrt(var*other_var)
    return corr


def make_cdf(img_flat):
    val, val_ct = np.unique(img_flat, return_counts=True)
    cdf = np.cumsum(val_ct)
    cdf = cdf/val_ct.sum()
    assert len(val) == len(cdf)
    assert cdf.max()<=1.0
    assert cdf.min()>=0.0
    return val, cdf


def step_from_processors(n_elements, n_processors, min_step):
    step = n_elements/(4*n_processors-1)
    if step < min_step:
        step = min_step
    return step


def _find_merger_candidates(roi_pair_list, dpix, output_list):
    local = []
    for pair in roi_pair_list:
        if do_rois_abut(pair[0], pair[1], dpix=dpix):
            local.append((pair[0].roi_id, pair[1].roi_id))
    for pair in local:
        output_list.append(pair)


def find_merger_candidates(roi_list: List[OphysROI],
                           dpix: float,
                           n_processors: int):
    mgr = multiprocessing.Manager()
    output_list = mgr.list()

    n_rois = len(roi_list)

    p_list = []

    n_pairs = n_rois*(n_rois-1)//2
    d_pairs = step_from_processors(n_rois, n_processors, 100)

    subset = []
    for i0 in range(n_rois):
        roi0 = roi_list[i0]
        for i1 in range(i0+1, n_rois, 1):
            roi1 = roi_list[i1]
            subset.append((roi0, roi1))
            if len(subset) >= d_pairs:
                args = (subset, dpix, output_list)
                p = multiprocessing.Process(target=_find_merger_candidates,
                                            args=args)
                p.start()
                p_list.append(p)
                subset = []
                while len(p_list) > 0 and len(p_list) >= (n_processors-1):
                    to_pop = []
                    for ii in range(len(p_list)-1, -1, -1):
                        if p_list[ii].exitcode is not None:
                            to_pop.append(ii)
                    for ii in to_pop:
                        p_list.pop(ii)

    if len(subset) > 0:
        args = (subset, dpix, output_list)
        p = multiprocessing.Process(target=_find_merger_candidates,
                                    args=args)
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    pair_list = [pair for pair in output_list]
    return pair_list
