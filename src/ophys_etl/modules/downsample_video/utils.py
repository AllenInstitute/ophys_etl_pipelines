from typing import Tuple, Optional
import pathlib
import h5py
import numpy as np

from ophys_etl.utils.array_utils import (
    downsample_array,
    n_frames_from_hz)

from ophys_etl.modules.median_filtered_max_projection.utils import (
    apply_median_filter_to_video)

import multiprocessing
import multiprocessing.managers
import time

import imageio
import tempfile


def _video_worker(
        input_path: pathlib.Path,
        input_hz: float,
        output_path: pathlib.Path,
        output_hz: float,
        kernel_size: int,
        input_slice: Tuple[int, int],
        output_lock: multiprocessing.managers.AcquirerProxy):
    t0 = time.time()
    frames_to_group = n_frames_from_hz(
            input_hz,
            output_hz)

    with h5py.File(input_path, 'r') as in_file:
        video_data = in_file['data'][input_slice[0]:input_slice[1], :, :]

    if frames_to_group > 1:
        video_data = downsample_array(video_data,
                                      input_fps=input_hz,
                                      output_fps=output_hz,
                                      strategy='average')

    video_data = apply_median_filter_to_video(video_data, kernel_size)
    start_index = input_slice[0] // frames_to_group
    end_index = start_index + video_data.shape[0]
    with output_lock:
        with h5py.File(output_path, 'a') as out_file:
            out_file['data'][start_index:end_index, :, :] = video_data
        duration = time.time()-t0
        print(f'completed chunk in {duration:.2e} seconds')

def create_downsampled_video_h5(
        input_path: pathlib.Path,
        input_hz: float,
        output_path: pathlib.Path,
        output_hz: float,
        kernel_size: int,
        n_processors: int):


    with h5py.File(input_path, 'r') as in_file:
        input_video_shape = in_file['data'].shape

    frames_to_group = n_frames_from_hz(
                            input_hz,
                            output_hz)

    n_frames_per_chunk = np.ceil(input_video_shape[0]/n_processors).astype(int)
    remainder = n_frames_per_chunk % frames_to_group
    n_frames_per_chunk += (frames_to_group-remainder)
    n_frames_per_chunk = max(1, n_frames_per_chunk)

    n_frames_out = np.ceil(input_video_shape[0]/frames_to_group).astype(int)
    with h5py.File(output_path, 'w') as out_file:
        out_file.create_dataset('data',
                                shape=(n_frames_out,
                                       input_video_shape[1],
                                       input_video_shape[2]),
                                chunks=(max(1, n_frames_out//100),
                                        input_video_shape[1],
                                        input_video_shape[2]),
                                dtype=float)

    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()
    process_list = []
    for i0 in range(0, input_video_shape[0], n_frames_per_chunk):
        print(f'starting {i0} -> {input_video_shape[0]}')
        p = multiprocessing.Process(
                target=_video_worker,
                args=(input_path,
                      input_hz,
                      output_path,
                      output_hz,
                      kernel_size,
                      (i0, i0+n_frames_per_chunk),
                      output_lock))
        p.start()
        process_list.append(p)
        #_video_worker(input_path, input_hz, output_path, output_hz,
        #              kernel_size, (i0, i0+n_frames_per_chunk),
        #              None)

    for p in process_list:
        p.join()



def _video_from_h5(
        h5_path: pathlib.Path,
        video_path: pathlib.Path,
        output_hz: int,
        quantiles: Optional[Tuple[float, float]] = None,
        reticle: bool = True,
        quality: int = 5):

    with h5py.File(h5_path, 'r') as in_file:
        video_shape = in_file['data'].shape
        full_data = in_file['data'][()]
        if quantiles is not None:
            q0, q1 = np.quantile(full_data, quantiles)
        else:
            q0 = full_data.min()
            q1 = full_data.max()
        del full_data
        print('got normalization')

        video_as_uint = np.zeros((video_shape[0],
                                  video_shape[1],
                                  video_shape[2],
                                  3), dtype=np.uint8)
        dt = 500
        for i0 in range(0, video_shape[0], dt):
            i1 = i0+dt
            data = in_file['data'][i0:i1, :, :].astype(float)
            if quantiles:
                data = np.where(data>q0, data, q0)
                data = np.where(data<q1, data, q1)
            data = np.round(255.0*(data-q0)/(q1-q0)).astype(np.uint8)
            for ic in range(3):
                video_as_uint[i0:i1, :, :, ic] = data

    print('constructed video_as_uint')

    if reticle:
        for ii in range(64, video_shape[1], 64):
            old_vals = np.copy(video_as_uint[:, ii:ii+2, :, :])
            new_vals = np.zeros(old_vals.shape, dtype=np.uint8)
            new_vals[:, :, :, 0] = 255
            new_vals = (new_vals//2) + (old_vals//2)
            new_vals = new_vals.astype(np.uint8)
            video_as_uint[:, ii:ii+2, :, :] = new_vals
        for ii in range(64, video_shape[2], 64):
            old_vals = np.copy(video_as_uint[:, :, ii:ii+2, :])
            new_vals = np.zeros(old_vals.shape, dtype=np.uint8)
            new_vals[:, :, :, 0] = 255
            new_vals = (new_vals//2) + (old_vals//2)
            new_vals = new_vals.astype(np.uint8)
            video_as_uint[:, :, ii:ii+2, :] = new_vals

    print('added reticles')

    imageio.mimsave(video_path,
                    video_as_uint,
                    fps=output_hz,
                    quality=quality,
                    pixelformat='yuv420p',
                    codec='libx264')
    print(f'wrote {video_path}')


def create_downsampled_video(
        input_path: pathlib.Path,
        input_hz: float,
        video_path: pathlib.Path,
        output_hz: float,
        kernel_size: int,
        n_processors: int,
        quality: int = 5,
        quantiles: Tuple[float, float] = (0.1, 0.99),
        reticle: bool = True,
        tmp_dir: Optional[pathlib.Path] = None):

    with tempfile.TemporaryDirectory(dir=tmp_dir) as this_tmp_dir:
        tmp_h5 = tempfile.mkstemp(dir=this_tmp_dir, suffix='.h5')[1]
        print(f'writing h5py to {tmp_h5}')

        create_downsampled_video_h5(
            input_path, input_hz,
            tmp_h5, output_hz,
            kernel_size,
            n_processors)

        print(f'wrote temp h5py to {tmp_h5}')

        _video_from_h5(
            tmp_h5,
            video_path,
            int(8*output_hz),
            quantiles=quantiles,
            quality=quality,
            reticle=reticle)
