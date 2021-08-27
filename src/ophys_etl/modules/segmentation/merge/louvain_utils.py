from typing import Tuple, Dict, Union, List
import numpy as np
import h5py
import pathlib
from itertools import combinations
import multiprocessing
import multiprocessing.managers
from ophys_etl.modules.segmentation.utils.multiprocessing_utils import (
    _winnow_process_list)


def update_merger_history(merger_history: Dict[int, int],
                          this_merger: Dict[str, int]) -> Dict[int, int]:
    """
    merger_history maps roi_id_in to roi_id_out
    this_merger has keys 'absorbed', 'absorber'

    WARNING will atler merger_history in-place, as well as return it
    """

    merger_history[this_merger['absorbed']] = this_merger['absorber']
    for roi_id_in in merger_history:
        if merger_history[roi_id_in] == this_merger['absorbed']:
            merger_history[roi_id_in] = this_merger['absorber']
    return merger_history


def _correlation_worker(
        sub_video: np.ndarray,
        filter_fraction: float,
        pixel_range: Tuple[int, int],
        lock: multiprocessing.managers.AcquirerProxy,
        output_file_path: pathlib.Path,
        dataset_name: str):
    """
    pixel_range is [min, max)
    """
    result = np.zeros((pixel_range[1]-pixel_range[0],
                       sub_video.shape[1]),
                      dtype=float)

    discard = 1.0-filter_fraction

    for i0 in range(pixel_range[0], pixel_range[1]):
        trace0 = sub_video[:, i0]
        th = np.quantile(trace0, discard)
        time0 = np.where(trace0 >= th)[0]
        for i1 in range(i0+1, sub_video.shape[1]):
            trace1 = sub_video[:, i1]
            th = np.quantile(trace1, discard)
            time1 = np.where(trace1 >= th)[0]
            time_mask = np.unique(np.concatenate([time0, time1]))
            f0 = trace0[time_mask]
            mu0 = np.mean(f0)
            var0 = np.var(f0, ddof=1)
            f1 = trace1[time_mask]
            mu1 = np.mean(f1)
            var1 = np.var(f1, ddof=1)
            num = np.mean((f0-mu0)*(f1-mu1))
            result[i0-pixel_range[0]][i1] = num/np.sqrt(var0*var1)

    with lock:
        with h5py.File(output_file_path, 'a') as output_file:
            output_file[dataset_name][pixel_range[0]:pixel_range[1],
                                      :] = result


def _correlate_all_pixels(
        sub_video: np.ndarray,
        filter_fraction,
        n_processors: int,
        scratch_file_path: pathlib.Path) -> np.ndarray:
    """
    result will just be upper-diagonal array
    """
    dataset_name = 'correlation'
    n_pixels = sub_video.shape[1]
    if n_pixels > 352:
        chunks = (352, 352)
    else:
        chunks = None
    with h5py.File(scratch_file_path, 'w') as out_file:
        out_file.create_dataset(dataset_name,
                                data=np.zeros((n_pixels, n_pixels),
                                              dtype=float),
                                chunks=None,
                                dtype=float)

    mgr = multiprocessing.Manager()
    lock = mgr.Lock()
    process_list = []
    d_pixels = n_pixels//(2*n_processors-2)
    for i0 in range(0, n_pixels, d_pixels):
        i1 = min(n_pixels, i0+d_pixels)
        p = multiprocessing.Process(target=_correlation_worker,
                                    args=(sub_video,
                                          filter_fraction,
                                          (i0, i1),
                                          lock,
                                          scratch_file_path,
                                          dataset_name))
        p.start()
        process_list.append(p)
        while len(process_list) > 0 and len(process_list) >= (n_processors-1):
            process_list = _winnow_process_list(process_list)
    for p in process_list:
        p.join()

    with h5py.File(scratch_file_path, 'r') as in_file:
        result = in_file[dataset_name][()]
    return result


def correlate_all_pixels(
        sub_video: np.ndarray,
        filter_fraction: float,
        n_processors: int,
        scratch_dir: pathlib.Path) -> np.ndarray:
    """
    sub_video: shape(n_time, n_pixels)
    result will be upper-diagonal array
    """
    scratch_file_path = None
    ii = 0
    while scratch_file_path is None:
        possible_fname = scratch_dir / f'correlation_array_{ii}.h5'
        if not possible_fname.is_file():
            scratch_file_path = possible_fname
        ii += 1
    try:
        result = _correlate_all_pixels(
                      sub_video,
                      filter_fraction,
                      n_processors,
                      scratch_file_path)
    finally:
        if scratch_file_path.exists():
            scratch_file_path.unlink()

    # fill in the entire matrix
    for i0 in range(result.shape[1]):
        result[i0+1:, i0] = result[i0, i0+1:]

    return result


def modularity(roi_id_arr: np.ndarray,
               pixel_corr: np.ndarray,
               weight_sum_arr: np.ndarray) -> float:
    """
    roi_id_arr: (n_pixels,) array of ROI IDs
    pixel_corr: (n_pixels, n_pixels) array of correlations
    weight_sum_arr: (n_pixels,) array that is np.sum(pixel_corr, axis=1)
    """
    weight_sum = 0.0
    for i0 in range(pixel_corr.shape[0]):
        weight_sum += pixel_corr[i0, i0+1:].sum()

    unique_roi_id = np.unique(roi_id_arr)
    qq = 0.0
    for roi_id in unique_roi_id:
        valid_roi_index = np.where(roi_id_arr==roi_id)[0]
        sub_corr = pixel_corr[valid_roi_index, :]
        sub_corr = sub_corr[:, valid_roi_index]
        sub_wgt_arr = weight_sum_arr[valid_roi_index]
        kk = np.outer(sub_wgt_arr, sub_wgt_arr)
        for ii in range(len(valid_roi_index)):
            kk[ii,ii] = 0.0
            sub_corr[ii, ii] = 0.0
        aa = sub_corr.sum()
        kk = kk.sum()
        qq += 0.5*(aa-(kk/(2.0*weight_sum)))
    return qq*0.5/weight_sum


def _louvain_clustering_iteration(
        roi_id_arr: np.ndarray,
        pixel_corr: np.ndarray,
        weight_sum_arr: np.ndarray) -> Tuple[bool,
                                            np.ndarray,
                                            Union[None, Dict[str, int]]]:
    """
    Find the one roi_id merger that maximizes modularity

    Just iterate over all possible pairs without regard to
    spatial distribution
    Returns
    -------
    has_changed: boolean
    roi_id: new roi_id array
    merger: Dict[str, int] absorber, absorbed
    """

    unq_roi_id = np.unique(roi_id_arr)
    best_roi_id_arr = roi_id_arr
    max_modularity = modularity(
                         roi_id_arr,
                         pixel_corr,
                         weight_sum_arr)

    has_changed = False
    merger = None
    for roi_id_pair in combinations(unq_roi_id, 2):
        # roi_id_pair is (absorber, absorbed)
        test_roi_id_arr = np.where(
                               roi_id_arr==roi_id_pair[1],
                               roi_id_pair[0],
                               roi_id_arr)

        new_modularity = modularity(
                            test_roi_id_arr,
                            pixel_corr,
                            weight_sum_arr)

        if new_modularity > max_modularity:
            has_changed = True
            max_modularity = new_modularity
            best_roi_id_arr = test_roi_id_arr
            merger = {'absorber': roi_id_pair[0],
                      'absorbed': roi_id_pair[1]}

    return (has_changed,
            best_roi_id_arr,
            merger)


