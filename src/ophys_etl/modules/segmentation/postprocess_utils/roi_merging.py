import matplotlib.figure as mplt_fig

import numpy as np
from typing import List, Tuple, Optional
import pathlib
import h5py
import time
import copy
import multiprocessing
import multiprocessing.managers
from sklearn.decomposition import PCA as sklearn_PCA

from scipy.spatial.distance import cdist
from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import get_roi_pixels
from ophys_etl.modules.segmentation.postprocess_utils.pdf_utils import (
    make_cdf,
    cdf_to_pdf,
    pdf_to_entropy)

import logging

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def _plot_mergers(img_arr: np.ndarray,
                 roi_lookup,
                 merger_pairs,
                 merger_metrics,
                 out_name):

    n = np.ceil(np.sqrt(len(merger_pairs))).astype(int)
    fig = mplt_fig.Figure(figsize=(n*7, n*7))
    axes = [fig.add_subplot(n,n,i) for i in range(1,len(merger_pairs)+1,1)]
    mx = img_arr.max()
    rgb_img = np.zeros((img_arr.shape[0], img_arr.shape[1], 3),
                       dtype=np.uint8)
    img = np.round(255*img_arr.astype(float)/max(1,mx)).astype(np.uint8)
    for ic in range(3):
        rgb_img[:,:,ic] = img
    del img

    alpha=0.5
    for ii in range(len(merger_pairs)):
        roi0 = roi_lookup[merger_pairs[ii][0]]
        roi1 = roi_lookup[merger_pairs[ii][1]]
        img = np.copy(rgb_img)

        for roi, color in zip((roi0, roi1),[(255,0,0),(0,255,0)]):
            msk = roi.mask_matrix
            for ir in range(roi.height):
                for ic in range(roi.width):
                    if not msk[ir, ic]:
                        continue
                    row = ir+roi.y0
                    col = ic+roi.x0
                    for jj in range(3):
                        old = rgb_img[row, col, jj]
                        new = np.round(alpha*color[jj]+(1.0-alpha)*old)
                        new = np.uint8(new)
                        img[row, col, jj] = new
            axes[ii].imshow(img)
            axes[ii].set_title('%.2e' % merger_metrics[ii], fontsize=15)
    for jj in range(ii, len(axes), 1):
        axes[jj].tick_params(left=0,bottom=0,labelleft=0,labelbottom=0)
        for s in ('top', 'left', 'bottom','right'):
            axes[jj].spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_name)


def plot_mergers(img_arr: np.ndarray,
                 roi_lookup,
                 merger_pairs,
                 merger_metrics,
                 out_name):

    n_sub = 16
    for i0 in range(0, len(merger_pairs), n_sub):
        s_pairs = merger_pairs[i0:i0+n_sub]
        s_metrics = merger_metrics[i0:i0+n_sub]
        _plot_mergers(img_arr, roi_lookup, s_pairs, s_metrics,
                      str(out_name).replace('.png',f'_{i0}.png'))


def _winnow_p_list(p_list):
    to_pop = []
    for ii in range(len(p_list)-1, -1, -1):
        if p_list[ii].exitcode is not None:
            to_pop.append(ii)
    for ii in to_pop:
        p_list.pop(ii)
    return p_list


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

    pixel mask is chosen from sub_video_0
    """

    assert sub_video_0.shape[0] == sub_video_1.shape[0]
    npix0 = sub_video_0.shape[1]
    npix1 = sub_video_1.shape[1]
    discard = max(0.0, 1.0-filter_fraction)

    corr = np.zeros((npix0, npix1), dtype=float)

    for i_pixel in range(npix0):
        raw_trace0 = sub_video_0[:, i_pixel]
        th = np.quantile(raw_trace0, discard)
        mask0 = (raw_trace0 > th)
        for i_other in range(npix1):
            trace1 = sub_video_1[:, i_other]
            th = np.quantile(trace1, discard)
            mask1 = (trace1 > th)
            mask = np.logical_or(mask0, mask1)
            trace0 = raw_trace0[mask]
            mu0 = np.mean(trace0)
            var0 = np.mean((trace0-mu0)**2)
            trace1 = trace1[mask]
            mu1 = np.mean(trace1)
            var1 = np.mean((trace1-mu1)**2)
            num = np.mean((trace0-mu0)*(trace1-mu1))
            denom = np.sqrt(var1*var0)
            corr[i_pixel, i_other] = num/denom

    assert corr.max()<=1.0
    assert corr.min()>=-1.0
    return corr



def step_from_processors(n_elements, n_processors,
                         min_step, denom_factor=4):
    step = n_elements//(denom_factor*n_processors-1)
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
                           unchanged_rois: Optional[set]=None,
                           n_processors: int = 8):
    mgr = multiprocessing.Manager()
    output_list = mgr.list()

    n_rois = len(roi_list)

    p_list = []

    n_pairs = n_rois*(n_rois-1)//2
    d_pairs = step_from_processors(n_pairs, n_processors, 100)

    subset = []
    for i0 in range(n_rois):
        roi0 = roi_list[i0]
        for i1 in range(i0+1, n_rois, 1):
            roi1 = roi_list[i1]
            if unchanged_rois is None:
                subset.append((roi0, roi1))
            else:
                if roi0.roi_id not in unchanged_rois or roi1.roi_id not in unchanged_rois:
                    subset.append((roi0, roi1))
            if len(subset) >= d_pairs:
                args = (copy.deepcopy(subset), dpix, output_list)
                p = multiprocessing.Process(target=_find_merger_candidates,
                                            args=args)
                p.start()
                p_list.append(p)
                subset = []
            while len(p_list) > 0 and len(p_list) >= (n_processors-1):
                p_list = _winnow_p_list(p_list)

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


def create_merger_video_lookup(sub_video_lookup: dict,
                               roi_pairs: list) -> dict:

    merger_video_lookup = {}
    for pair in roi_pairs:
        n0 = sub_video_lookup[pair[0]].shape[1]
        nt = sub_video_lookup[pair[0]].shape[0]
        n1 = sub_video_lookup[pair[1]].shape[1]
        assert sub_video_lookup[pair[1]].shape[0] == nt
        new_sub_video = np.zeros((nt, n0+n1), dtype=float)
        new_sub_video[:, :n0] = sub_video_lookup[pair[0]]
        new_sub_video[:, n0:] = sub_video_lookup[pair[1]]
        merger_video_lookup[pair] = new_sub_video
    return merger_video_lookup


def create_sub_video_lookup(video_data: np.ndarray,
                            roi_list: List[OphysROI]) -> dict:
    """
    Video is not flattened in space; output will be
    flattened in space
    """
    sub_video_lookup = {}
    for roi in roi_list:
        sub_video = video_data[:,
                               roi.y0:roi.y0+roi.height,
                               roi.x0:roi.x0+roi.width]

        sub_video = sub_video.reshape(sub_video.shape[0], -1)
        roi_mask = roi.mask_matrix.flatten()
        sub_video_lookup[roi.roi_id] = sub_video[:, roi_mask]
        sub_video_lookup[roi.roi_id] = sub_video

    return sub_video_lookup


def _self_corr_subset(roi_id_list: List[int],
                      sub_video_lookup: dict,
                      filter_fraction:float,
                      output_dict):

    local_dict = {}
    for roi_id in roi_id_list:
        sub_video = sub_video_lookup[roi_id]
        corr = correlate_sub_videos(sub_video,
                                    sub_video,
                                    filter_fraction)

        assert corr.shape[0] == corr.shape[1]
        #mask = np.ones(corr.shape, dtype=bool)
        #for ii in range(corr.shape[0]):
        #    for jj in range(ii+1):
        #       mask[ii,jj] = False
        #corr = corr[mask].flatten()

        corr = np.mean(corr, axis=1)
        local_dict[roi_id] = corr

    k_list = list(local_dict.keys())
    for k in k_list:
        output_dict[k] = local_dict.pop(k)


def create_self_correlation_lookup(roi_list: List[OphysROI],
                                   sub_video_lookup: dict,
                                   filter_fraction: float,
                                   n_processors: int,
                                   shuffler: np.random.RandomState):

    roi_id_list = [roi.roi_id for roi in roi_list]
    shuffler.shuffle(roi_id_list)

    n_rois = len(roi_list)
    d_roi = step_from_processors(n_rois, n_processors, 1)
    print('self corr step ',d_roi)

    mgr = multiprocessing.Manager()
    output_dict = mgr.dict()
    p_list = []
    for i0 in range(0, n_rois, d_roi):
        subset = roi_id_list[i0: i0+d_roi]
        args = (subset,
                sub_video_lookup,
                filter_fraction,
                output_dict)
        p = multiprocessing.Process(target=_self_corr_subset,
                                    args=args)
        p.start()
        p_list.append(p)
        if len(p_list) > 0 and len(p_list) >= (n_processors-1):
            p_list = _winnow_p_list(p_list)
    for p in p_list:
        p.join()
    final_output = {}
    k_list = list(output_dict.keys())
    for k in k_list:
        final_output[k] = output_dict.pop(k)
    return final_output


def calculate_merger_chisq(large_self_corr: np.ndarray,
                          cross_corr: np.ndarray):
    """
    Take the self correlation of a large ROI and
    the cross-correlation between a smaller ROI and
    the larger ROI. Find the probability that the
    cross correlation values fall within the
    CDF of the larger cross correlations. Sum those
    to make a chisquared. Return chisq per dof
    """
    (cdf_bins,
     cdf_vals) = make_cdf(large_self_corr)

    prob = np.interp(cross_corr, cdf_bins, cdf_vals,
                     left=0.0, right=1.0)

    eps = 1.0e-6
    prob = np.where(prob>eps, prob, eps)
    chisq = -2.0*np.log(prob).sum()
    chisq_per_dof = chisq/len(cross_corr)

    return chisq_per_dof


def _calc_entropy(corr: np.ndarray):
    if len(corr) == 0:
        return 0.0
    dx = 0.01
    hist = np.round(corr/dx).astype(int)
    unq, unq_ct = np.unique(hist, return_counts=True)
    unq_ct = unq_ct.astype(float)
    tot = unq_ct.sum()
    prob = unq_ct/tot
    return -1.0*np.sum(prob*np.log(prob))


def _chisq_from_video(sub_video, n_components=3):
    """
    sub_video is (nt, npix)
    """
    if sub_video.shape[1] < (n_components+1):
        return 0.0
    npix = sub_video.shape[1]
    ntime = sub_video.shape[0]
    sub_video = sub_video.T

    pca = sklearn_PCA(n_components=n_components, copy=True)

    transformed_video = pca.fit_transform(sub_video)
    assert transformed_video.shape==(sub_video.shape[0], n_components)

    mu = np.mean(transformed_video, axis=0)
    assert mu.shape == (n_components,)
    distances = np.sqrt(((transformed_video-mu)**2).sum(axis=1))
    assert distances.shape == (npix,)
    std = np.std(distances, ddof=1)
    chisq = ((transformed_video-mu)/std)**2
    chisq = chisq.sum()

    return chisq+2.0*n_components*npix*np.log(std)+n_components*npix*np.log(2.0*np.pi)


def _evaluate_merger_subset(roi_pair_list: List[Tuple[int, int]],
                            sub_video_lookup: dict,
                            merger_video_lookup: dict,
                            filter_fraction: float,
                            p_value: float,
                            output_dict):

    local_output = {}
    n_components=3

    for pair in roi_pair_list:

        video0 = sub_video_lookup[pair[0]]
        video1 = sub_video_lookup[pair[1]]
        merger_video = merger_video_lookup[pair]
        npix = merger_video.shape[1]
        npix0 = video0.shape[1]
        npix1 = video1.shape[1]
        assert npix == (npix0+npix1)

        chisq0 = _chisq_from_video(video0,
                                   n_components=n_components)

        chisq1 = _chisq_from_video(video1,
                                   n_components=n_components)

        chisq_merger = _chisq_from_video(merger_video,
                                         n_components=n_components)

        # PCA is effectively finding the three center components
        # and the three sigmas
        bic_baseline = 4*n_components*np.log(npix) + chisq0 + chisq1
        bic_merger = 2*n_components*np.log(npix) + chisq_merger
        d_bic = bic_merger-bic_baseline

        if d_bic < 0.0:
            print(f'd_bic {d_bic} chisq_m {chisq_merger} chisq0 {chisq0} chisq1 {chisq1} '
                  f'pixels {video0.shape[1]} {video1.shape[1]}')
            local_output[(pair[0], pair[1])] = d_bic

    k_list = list(local_output.keys())
    for k in k_list:
        output_dict[k] = local_output.pop(k)


def evaluate_mergers(roi_pair_list: List[Tuple[int, int]],
                     sub_video_lookup: dict,
                     merger_video_lookup: dict,
                     filter_fraction: float,
                     p_value: float,
                     n_processors: int,
                     shuffler: np.random.RandomState):

    t0 = time.time()
    roi_pair_list = copy.deepcopy(roi_pair_list)
    shuffler.shuffle(roi_pair_list)

    mgr = multiprocessing.Manager()
    output_dict = mgr.dict()

    n_pairs = len(roi_pair_list)
    d_pairs = step_from_processors(n_pairs, n_processors, 2,
                                   denom_factor=2)

    p_list = []
    for i0 in range(0, n_pairs, d_pairs):
        subset = roi_pair_list[i0:i0+d_pairs]
        args = (subset,
                sub_video_lookup,
                merger_video_lookup,
                filter_fraction,
                p_value,
                output_dict)
        p = multiprocessing.Process(target=_evaluate_merger_subset,
                                    args=args)
        p.start()
        p_list.append(p)
        while len(p_list) > 0 and len(p_list) >= (n_processors-1):
            p_list = _winnow_p_list(p_list)

    for p in p_list:
        p.join()

    final_output = {}
    k_list = list(output_dict.keys())
    for k in k_list:
        final_output[k] = output_dict.pop(k)
    return final_output


def attempt_merger_pixel_correlation(video_data: np.ndarray,
                                     roi_list: List[OphysROI],
                                     filter_fraction: float,
                                     shuffler: np.random.RandomState,
                                     n_processors: int,
                                     unchanged_roi: set,
                                     img_data=None,
                                     i_pass:int = 0,
                                     diagnostic_dir=None):

    did_a_merger = False
    roi_lookup = {}
    for roi in roi_list:
        if roi.roi_id in roi_lookup:
            raise RuntimeError(f'roi_id {roi.roi_id} duplicated in '
                               'attempt_merger_pixel_correlation')

        roi_lookup[roi.roi_id] = roi

    p_value = 0.05

    t0 = time.time()
    sub_video_lookup = create_sub_video_lookup(video_data, roi_list)
    logger.info(f'created sub_video_lookup in {time.time()-t0:.2f} seconds')

    t0 = time.time()
    merger_candidates = find_merger_candidates(roi_list,
                                               np.sqrt(2.0),
                                               unchanged_rois=None,
                                               n_processors=n_processors)
    logger.info('found merger_candidates '
                f'({len(merger_candidates)} {len(unchanged_roi)})'
                f' in {time.time()-t0:.2f} seconds')

    #for pair in merger_candidates:
    #    assert (pair[0] not in unchanged_roi or pair[1] not in unchanged_roi)

    t0 = time.time()
    merger_video_lookup = create_merger_video_lookup(sub_video_lookup,
                                                     merger_candidates)
    logger.info(f'created merger video lookup in {time.time()-t0:.2f} seconds')

    t0 = time.time()
    mergers = evaluate_mergers(merger_candidates,
                               sub_video_lookup,
                               merger_video_lookup,
                               filter_fraction,
                               p_value,
                               n_processors,
                               shuffler)
    logger.info(f'evaluated mergers in {time.time()-t0:.2f} seconds')
    logger.info(f'found {len(mergers)} potential mergers')

    merger_values = []
    merger_pairs = []
    k_list = list(mergers.keys())
    for k in k_list:
        merger_pairs.append(k)
        merger_values.append(mergers.pop(k))
    merger_values = np.array(merger_values)
    merger_pairs = np.array(merger_pairs)
    sorted_indexes = np.argsort(merger_values)
    merger_values = merger_values[sorted_indexes]
    merger_pairs = merger_pairs[sorted_indexes]

    if img_data is None:
        img_data = np.zeros(video_data.shape[1:], dtype=np.uint8)

    new_roi_lookup = {}
    has_been_considered = set()
    has_been_merged = set()

    incoming_rois = list(roi_lookup.keys())
    been_merged_pairs = []
    been_merged_lookup = {}
    been_merged_values = []

    for pair, vv in zip(merger_pairs, merger_values):

        roi_id_0 = pair[0]
        roi_id_1 = pair[1]

        if roi_id_0 == roi_id_1:
            continue

        keep_going = True
        if roi_id_0 in has_been_considered or roi_id_1 in has_been_considered:
            keep_going = False

        has_been_considered.add(roi_id_0)
        has_been_considered.add(roi_id_1)

        if not keep_going:
            continue

        if roi_id_0 in has_been_merged or roi_id_1 in has_been_merged:
            continue

        roi0 = roi_lookup[roi_id_0]
        roi1 = roi_lookup[roi_id_1]

        # if candidate ROIs abut a merger,
        # do not consider them (in case they
        # would be a better fit in the new
        # merger)
        for roi_id in new_roi_lookup:
            _new_roi = new_roi_lookup[roi_id]
            if do_rois_abut(roi0, _new_roi):
                keep_going = False
                break
            if do_rois_abut(roi1, _new_roi):
                keep_going = False
                break

        if not keep_going:
            continue

        if roi0.mask_matrix.sum() > roi1.mask_matrix.sum():
            new_roi_id = roi0.roi_id
        else:
            new_roi_id = roi1.roi_id

        been_merged_pairs.append((roi_id_0, roi_id_1))
        been_merged_lookup[roi_id_0] = roi0
        been_merged_lookup[roi_id_1] = roi1
        been_merged_values.append(vv)

        new_roi = merge_rois(roi0, roi1, new_roi_id)

        if new_roi.roi_id in new_roi_lookup:
            raise RuntimeError(f'roi_id {new_roi.roi_id} '
                               'duplicated in new lookup')

        new_roi_lookup[new_roi.roi_id] = new_roi
        did_a_merger = True
        has_been_merged.add(roi_id_0)
        has_been_merged.add(roi_id_1)

    if diagnostic_dir is not None:
        accepted_file = diagnostic_dir / f'accepted_mergers_{i_pass}.png'
        plot_mergers(img_data, been_merged_lookup, been_merged_pairs, been_merged_values,
                     accepted_file)


    unchanged_roi = set()
    for roi_id in roi_lookup:
        if roi_id in has_been_merged:
            continue
        if roi_id in new_roi_lookup:
            raise RuntimeError(f'on final pass roi_id {roi_id} '
                               'duplicated in new lookup')
        new_roi_lookup[roi_id] = roi_lookup[roi_id]
        unchanged_roi.add(roi_id)

    for roi_id in incoming_rois:
        if roi_id not in unchanged_roi and roi_id not in has_been_merged:
            raise RuntimeError(f'lost track of {roi_id} in merging')

    return (did_a_merger,
           list(new_roi_lookup.values()),
           unchanged_roi)
