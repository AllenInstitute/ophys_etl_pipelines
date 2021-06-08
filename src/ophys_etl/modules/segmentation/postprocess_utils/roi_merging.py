import numpy as np
from typing import List, Tuple

from scipy.spatial.distance import cdist
from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import get_roi_pixels


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

    distances = cdist(array_0, array_1, metric='euclidean')
    if distances.min() <= dpix:
        return True
    return False


def trace_from_array(video: np.ndarray,
                     roi: OphysROI) -> np.ndarray:

    xmin = roi.x0
    ymin = roi.y0
    xmax = roi.x0+roi.width
    ymax = roi.y0+roi.height

    sub_video = video[:, xmin:xmax, ymin:ymax]
    sub_video = sub_video.reshape(sub_video.shape[0], -1)
    mask_flat = roi.mask_matrix.flatten()

    npix = mask_flat.sum()
    trace = sub_video[:, mask_flat].sum(axis=1)/npix
    assert trace.shape == (sub_video.shape[0],)
    return trace


def correlate_traces(trace0: np.ndarray,
                     trace1: np.ndarray,
                     filter_fraction: float) -> float:

    discard = 1.0-filter_fraction
    if discard > 0.0:
        th0 = np.quantile(trace0, discard)
        mask0 = (trace0 > th0)
        th1 = np.quantile(trace1, discard)
        mask1 = (trace1 > th1)
        mask = np.logical_or(mask0, mask1)
        trace0 = trace0[mask]
        trace1 = trace1[mask]

    mu0 = np.mean(trace0)
    var0 = np.mean((trace0-mu0)**2)
    mu1 = np.mean(trace1)
    var1 = np.mean((trace1-mu1)**2)
    return np.mean((trace0-mu0)*(trace1-mu1))/np.sqrt(var0*var1)


def attempt_merger(video: np.ndarray,
                   roi_list: List[OphysROI],
                   filter_fraction: float,
                   threshold: float) -> Tuple[bool, List[OphysROI]]:

    pairs = []
    metric_values = []

    n_roi = len(roi_list)
    n_time = video.shape[0]
    traces = np.zeros((n_roi, n_time), dtype=float)
    roi_lookup = {}
    for i_roi in range(n_roi):
        traces[i_roi, :] = trace_from_array(video, roi_list[i_roi])
        roi_lookup[i_roi] = roi_list[i_roi]

    for i0 in range(n_roi):
        roi0 = roi_list[i0]
        for i1 in range(i0+1, n_roi):
            roi1 = roi_list[i1]
            if not do_rois_abut(roi0, roi1, dpix=np.sqrt(2)):
                continue
            pearson = correlate_traces(traces[i0,:],
                                       traces[i1,:],
                                       filter_fraction=filter_fraction)

            if pearson >= threshold:
                pairs.append((i0, i1))
                metric_values.append(pearson)

    if len(metric_values) == 0:
        return False, roi_list

    # sort potential mergers by correlation
    metric_values = np.array(metric_values)
    sorted_indices = np.argsort(-1.0*metric_values)

    new_roi_list = []
    has_merged = False
    for metric_index in sorted_indices:
        candidate_pair = pairs[metric_index]
        if candidate_pair[0] in roi_lookup and candidate_pair[1] in roi_lookup:
            roi0 = roi_lookup.pop(candidate_pair[0])
            roi1 = roi_lookup.pop(candidate_pair[1])
            if roi0.mask_matrix.sum() > roi1.mask_matrix.sum():
                new_roi_id = roi0.roi_id
            else:
                new_roi_id = roi1.roi_id
            new_roi = merge_rois(roi0, roi1, new_roi_id)
            new_roi_list.append(new_roi)
            has_merged = True

    for roi_id in roi_lookup:
        new_roi_list.append(roi_lookup[roi_id])
    return has_merged, new_roi_list


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


def sub_video_from_roi(video: np.ndarray,
                       roi: OphysROI) -> np.ndarray:
    """
    Video is not flattened in space; output will be
    flattened in space
    """
    sub_video = video[:,
                      roi.y0:roi.y0+roi.height,
                      roi.x0:roi.x0+roi.width]

    sub_video = sub_video.reshape(sub_video.shape[0], -1)
    roi_mask = roi.mask_matrix.flatten()
    return sub_video[:, roi_mask]

def attempt_merger_pixel_correlation(
                   video: np.ndarray,
                   roi_list: List[OphysROI],
                   filter_fraction: float,
                   threshold: float) -> Tuple[bool, List[OphysROI]]:

    n_roi = len(roi_list)
    merger_candidates = []
    merger_goodness = []
    for i0 in range(n_roi):
        roi0 = roi_list[i0]
        sub_video_0 = sub_video_from_roi(video, roi0)
        self_corr_0 = correlate_sub_videos(sub_video_0,
                                           sub_video_0,
                                           filter_fraction)
        assert self_corr_0.shape[0] == self_corr_0.shape[1]
        mask = np.ones(self_corr_0.shape, dtype=bool)
        for ii in range(self_corr_0.shape[0]):
            mask[ii,ii] = False
        self_corr_0 = self_corr_0[mask].flatten()
        mu0 = np.mean(self_corr_0)
        std0 = np.std(self_corr_0, ddof=1)

        for i1 in range(i0+1, n_roi):
            roi1 = roi_list[i1]
            if not do_rois_abut(roi0, roi1, dpix=np.sqrt(2)):
                continue
            sub_video_1 = sub_video_from_roi(video, roi1)
            self_corr_1 = correlate_sub_videos(sub_video_1,
                                               sub_video_1,
                                               filter_fraction)

            assert self_corr_1.shape[0] == self_corr_1.shape[1]

            mask = np.ones(self_corr_1.shape, dtype=bool)
            for ii in range(self_corr_1.shape[0]):
                mask[ii,ii] = False
            self_corr_1 = self_corr_1[mask].flatten()
            mu1 = np.mean(self_corr_1)
            std1 = np.std(self_corr_1, ddof=1)

            if len(self_corr_0) > len(self_corr_1):
                big_self = self_corr_0
                small_self = self_corr_1
                mu = mu0
                std = std0
                cross = correlate_sub_videos(sub_video_1,
                                             sub_video_0,
                                             filter_fraction)
            else:
                big_self = self_corr_1
                small_self = self_corr_0
                mu = mu1
                std = std1
                cross = correlate_sub_videos(sub_video_0,
                                             sub_video_1,
                                             filter_fraction)
            cross = cross.flatten()
            mu_cross = np.mean(cross)
            std_cross = np.std(cross, ddof=1)
            dist = np.abs(mu_cross-mu)
            if dist < (std+std_cross):
                merger_candidates.append((i0, i1))
                merger_goodness.append(dist/(std+std_cross))

    if len(merger_candidates) == 0:
        return False, roi_list

    has_merged = False
    merger_goodness = np.array(merger_goodness)
    sorted_indices = np.argsort(merger_goodness)
    merged_rois = set()
    output_list = []
    for i_merger in sorted_indices:
        candidate = merger_candidates[i_merger]
        i0 = candidate[0]
        i1 = candidate[1]
        if i0 in merged_rois or i1 in merged_rois:
            continue
        roi0 = roi_list[i0]
        roi1 = roi_list[i1]
        if roi0.mask_matrix.sum()>roi1.mask_matrix.sum():
            new_id = roi0.roi_id
        else:
            new_id = roi1.roi_id
        new_roi = merge_rois(roi0, roi1, new_id)
        has_merged = True
        output_list.append(new_roi)
        merged_rois.add(i0)
        merged_rois.add(i1)

    for i_roi in range(len(roi_list)):
        if i_roi in merged_rois:
            continue
        output_list.append(roi_list[i_roi])
    return has_merged, output_list
