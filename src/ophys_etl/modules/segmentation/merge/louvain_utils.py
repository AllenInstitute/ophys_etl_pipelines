from typing import Tuple, Dict, Union, List, Optional, Set
import numpy as np
import h5py
import time
import pathlib
import networkx
from itertools import combinations
import multiprocessing
import multiprocessing.managers

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.merge.candidates import (
    update_neighbor_lookup)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    do_rois_abut)

from ophys_etl.modules.segmentation.utils.multiprocessing_utils import (
    _winnow_process_list)


def find_roi_clusters(
        roi_list: List[OphysROI],
        pixel_distance: float = np.sqrt(2.0)) -> List[List[OphysROI]]:
    """
    Given a list of OphysROI, find a list of lists representing
    the complexes of contiguous ROIs
    """

    roi_lookup = {roi.roi_id: roi for roi in roi_list}
    graph = networkx.Graph()
    for i0 in range(len(roi_list)):
        graph.add_node(roi_list[i0].roi_id)
        for i1 in range(i0+1, len(roi_list)):
            if do_rois_abut(roi_list[i0], roi_list[i1], pixel_distance):
                graph.add_edge(roi_list[i0].roi_id, roi_list[i1].roi_id)
    output = []
    for component in networkx.connected_components(graph):
        local_list = []
        for roi_id in component:
            local_list.append(roi_lookup[roi_id])
        output.append(local_list)
    return output


def update_merger_history(merger_history: Dict[int, int],
                          this_merger: Dict[str, int]) -> Dict[int, int]:
    """
    merger_history maps roi_id_in to roi_id_out
    this_merger has keys 'absorbed', 'absorber'

    WARNING will alter merger_history in-place, as well as return it
    """

    merger_history[this_merger['absorbed']] = this_merger['absorber']
    for roi_id_in in merger_history:
        if merger_history[roi_id_in] == this_merger['absorbed']:
            merger_history[roi_id_in] = this_merger['absorber']
    return merger_history


def _correlation_worker(
        sub_video: np.ndarray,
        pixel_distances: Union[np.ndarray, None],
        kernel_size: Union[float, None],
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

    pixel_index_array = np.arange(sub_video.shape[1])

    for i0 in range(pixel_range[0], pixel_range[1]):
        trace0 = sub_video[:, i0]
        th = np.quantile(trace0, discard)
        time0 = np.where(trace0 >= th)[0]
        if kernel_size is not None:
            other_pixel_mask = np.logical_and(
                                  pixel_index_array>i0,
                                  pixel_distances[i0,:]<=kernel_size)
        else:
            other_pixel_mask = pixel_index_array > i0
        other_pixel_indices = pixel_index_array[other_pixel_mask]
        for i1 in other_pixel_indices:
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
        pixel_distances: Union[np.ndarray, None],
        kernel_size: Union[float, None],
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
    d_pixels = max(1, n_pixels//(2*n_processors-2))
    for i0 in range(0, n_pixels, d_pixels):
        i1 = min(n_pixels, i0+d_pixels)
        p = multiprocessing.Process(target=_correlation_worker,
                                    args=(sub_video,
                                          pixel_distances,
                                          kernel_size,
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
        scratch_dir: pathlib.Path,
        pixel_distances: Optional[np.ndarray] = None,
        kernel_size: Optional[float] = None) -> np.ndarray:
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
                      pixel_distances,
                      kernel_size,
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
        weight_sum_arr: np.ndarray,
        valid_pairs: Optional[List[Tuple[int, int]]] = None) -> Dict:
    """
    Find the one roi_id merger that maximizes modularity

    Just iterate over all possible pairs without regard to
    spatial distribution
    Returns
    -------
    has_changed: boolean
    roi_id: new roi_id array
    merger: Dict[str, int] absorber, absorbed
    best_modularity
    """

    unq_roi_id = np.unique(roi_id_arr)
    best_roi_id_arr = roi_id_arr
    max_modularity = modularity(
                         roi_id_arr,
                         pixel_corr,
                         weight_sum_arr)

    has_changed = False
    merger = None

    if valid_pairs is None:
        valid_pairs = list(combinations(unq_roi_id, 2))

    for roi_id_pair in valid_pairs:
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

    return {'has_changed': has_changed,
            'roi_id_arr': best_roi_id_arr,
            'this_merger': merger,
            'modularity': max_modularity}


def _louvain_clustering_worker(
        roi_id_arr: np.ndarray,
        pixel_corr: np.ndarray,
        weight_sum_arr: np.ndarray,
        valid_pairs: List[Tuple[int, int]],
        process_id: int,
        output_dict: multiprocessing.managers.DictProxy) -> None:
    """
    """

    result = _louvain_clustering_iteration(
                roi_id_arr,
                pixel_corr,
                weight_sum_arr,
                valid_pairs=valid_pairs)

    output_dict[process_id] = result


def _do_louvain_clustering(
      roi_id_arr: np.ndarray,
      pixel_corr: np.ndarray,
      neighbor_lookup: Optional[Dict[int, Set[int]]] = None,
      n_processors: Optional[int] = None) -> Tuple[np.ndarray,
                                                   List[Dict[str, Tuple[int]]]]:
    """
    index_to_pixel_coords: maps i_pixel to (row, col)
    roi_id_arr: maps i_pixel to roi_id
    pixel_corr: (n_pixels, n_pixels) correlation

    Returns
    -------
    roi_id_arr
    List[Dict[new_roi_id, Tuple of absorbed ROI IDs]]
    """

    # set any correlations < 0 to 0
    pixel_corr = np.where(pixel_corr>=0.0,
                          pixel_corr,
                          0.0)

    n_pixels = pixel_corr.shape[0]

    weight_sum_arr = np.sum(pixel_corr, axis=1)
    # exclude self correlation
    for ii in range(n_pixels):
        weight_sum_arr[ii] -= pixel_corr[ii, ii]

    # maps input ROI ID to output ROI ID
    final_mergers = dict()
    for roi_id in roi_id_arr:
        final_mergers[roi_id] = roi_id

    keep_going = True
    _n0 = len(np.unique(roi_id_arr))
    _t0 = time.time()

    while keep_going:
        _n_proc = 0
        if neighbor_lookup is None:
            valid_pairs = list(combinations(np.unique(roi_id_arr), 2))
        else:
            valid_pairs = set()
            for i0 in neighbor_lookup:
                for i1 in neighbor_lookup[i0]:
                    if i0 > i1:
                        t = (i0, i1)
                    else:
                        t = (i1, i0)
                    valid_pairs.add(t)
            valid_pairs = list(valid_pairs)
        _n_valid = len(valid_pairs)

        mgr = multiprocessing.Manager()
        output_dict = mgr.dict()
        process_list = []
        d_pairs = max(1,
                      np.round(0.6666*_n_valid/n_processors).astype(int))

        for i0 in range(0, _n_valid, d_pairs):
            subset = valid_pairs[i0:i0+d_pairs]
            p = multiprocessing.Process(
                        target=_louvain_clustering_worker,
                        args=(roi_id_arr,
                              pixel_corr,
                              weight_sum_arr,
                              subset,
                              i0,
                              output_dict))

            p.start()
            _n_proc += 1
            process_list.append(p)
            while len(process_list) > 0 and len(process_list) >= (n_processors-1):
                process_list = _winnow_process_list(process_list)

        for p in process_list:
            p.join()

        keep_going = False
        this_merger = None
        max_modularity = None
        for pid in output_dict:
            candidate = output_dict[pid]
            if not candidate['has_changed']:
                continue
            if max_modularity is None or candidate['modularity'] > max_modularity:
                max_modularity = candidate['modularity']
                keep_going = True
                roi_id_arr = candidate['roi_id_arr']
                this_merger = candidate['this_merger']

        _n_roi = len(np.unique(roi_id_arr))
        _duration = time.time()-_t0
        print(f'{_n_roi} from {_n0} ROI after '
              f'{_duration:.2e} seconds; '
              f'considered {_n_valid} pairs; '
              f'used {_n_proc} processes')

        if this_merger is not None:
            final_mergers = update_merger_history(
                                  final_mergers,
                                  this_merger)
            if neighbor_lookup is not None:
                neighbor_lookup = update_neighbor_lookup(
                                      neighbor_lookup,
                                      [this_merger])

    return (roi_id_arr, final_mergers)
