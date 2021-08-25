from typing import Tuple
import numpy as np
import h5py
import time
import pathlib
import multiprocessing
import multiprocessing.managers
from ophys_etl.modules.segmentation.utils.multiprocessing_utils import (
    _winnow_process_list)


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
    print(f'starting {pixel_range}')
    result = np.zeros((pixel_range[1]-pixel_range[0],
                       sub_video.shape[1]),
                      dtype=float)
    t0 = time.time()
    ct = 0
    discard = 1.0-filter_fraction
    tot = (pixel_range[1]-pixel_range[0])*sub_video.shape[1]
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
            ct += 1
            if ct % 10000 == 0:
                duration = time.time()-t0
                per = duration/ct
                estimate = per*tot
                remain = estimate-duration
                print(f'{ct} in {duration:.2e} -- {per:.2e} {remain:.2e} {estimate:.2e}')

    with lock:
        print(f'writing {pixel_range}')
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
    print(f'creating {scratch_file_path}')
    with h5py.File(scratch_file_path, 'w') as out_file:
        out_file.create_dataset(dataset_name,
                                data = np.zeros((n_pixels, n_pixels), dtype=float),
                                chunks=(352, 352),
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
        while len(process_list) >0 and len(process_list) >= (n_processors-1):
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
            print(f'unlinking {scratch_file_path}')
            scratch_file_path.unlink()

    return result


