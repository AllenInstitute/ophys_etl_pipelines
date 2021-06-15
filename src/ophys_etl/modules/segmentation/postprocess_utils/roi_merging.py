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

        npix = roi0.mask_matrix.sum()+roi1.mask_matrix.sum()

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
            per_dof = merger_metrics[ii]/npix
            axes[ii].set_title('%.2e (%.2e per pix)' % (merger_metrics[ii], per_dof),
                              fontsize=15)
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


def _chisq_from_video(sub_video,
                      fit_video,
                      n_components=3):
    """
    sub_video is (nt, npix)
    """
    if sub_video.shape[1] < (n_components+1):
        return 0.0
    npix = sub_video.shape[1]
    ntime = sub_video.shape[0]
    sub_video = sub_video.T

    pca = sklearn_PCA(n_components=n_components, copy=True)

    pca.fit(fit_video)
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

        if npix0>npix1:
            fit_video = video0
        else:
            fit_video = video1

        chisq0 = _chisq_from_video(video0, video0,
                                   n_components=n_components)

        chisq1 = _chisq_from_video(video1, video1,
                                   n_components=n_components)

        chisq_merger = _chisq_from_video(merger_video, merger_video,
                                         n_components=n_components)

        # PCA is effectively finding the three center components
        # and the three sigmas
        bic_baseline = 4*n_components*np.log(npix) + chisq0 + chisq1
        bic_merger = 2*n_components*np.log(npix) + chisq_merger
        d_bic = bic_merger-bic_baseline

        if d_bic < 0.0:
            print(f'd_bic/dof {d_bic/npix} chisq_m {chisq_merger} chisq0 {chisq0} chisq1 {chisq1} '
                  f'pixels {video0.shape[1]} {video1.shape[1]}')
            local_output[(pair[0], pair[1])] = d_bic/npix

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
