from typing import Optional, Dict, List
import tifffile
import h5py
import pathlib
import numpy as np
import tempfile
import os


def split_timeseries_tiff(
        tiff_path: pathlib.Path,
        offset_to_path: Dict,
        tmp_dir: Optional[pathlib.Path] = None,
        dump_every: int = 1000):

    offset_to_tmp_files = dict()
    offset_to_tmp_dir = dict()
    for offset in offset_to_path:
        pth = offset_to_path[offset]
        if tmp_dir is not None:
            offset_to_tmp_dir[offset] = tmp_dir
        else:
            offset_to_tmp_dir[offset] = pth.parent
        offset_to_tmp_files[offset] = []

    try:
        _split_timeseries_tiff(
            tiff_path=tiff_path,
            offset_to_path=offset_to_path,
            offset_to_tmp_files=offset_to_tmp_files,
            offset_to_tmp_dir=offset_to_tmp_dir,
            dump_every=dump_every)
    finally:
        for offset in offset_to_tmp_files:
            for tmp_pth in offset_to_tmp_files[offset]:
                if tmp_pth.exists():
                    os.unlink(tmp_pth)


def _split_timeseries_tiff(
        tiff_path: pathlib.Path,
        offset_to_path: Dict,
        offset_to_tmp_files: Dict,
        offset_to_tmp_dir: Dict,
        dump_every: int = 1000):

    max_offset = max(list(offset_to_path.keys()))

    fov_shape = None
    video_dtype = None
    offset_to_cache = dict()
    offset_to_valid_cache = dict()

    with tifffile.TiffFile(tiff_path, mode='rb') as tiff_file:
        current_offset = -1
        cache_ct = 0
        for page in tiff_file.pages:
            arr = page.asarray()

            if fov_shape is None:
                fov_shape = arr.shape
                video_dtype = arr.dtype
                for offset in offset_to_path:
                    cache = np.zeros((dump_every,
                                      fov_shape[0],
                                      fov_shape[1]),
                                     dtype=video_dtype)
                    offset_to_cache[offset] = cache

            current_offset += 1
            if current_offset > max_offset:
                current_offset = 0
                cache_ct += 1
                if cache_ct == dump_every:
                    _dump_timeseries_caches(
                        offset_to_cache=offset_to_cache,
                        offset_to_valid_cache=offset_to_valid_cache,
                        offset_to_tmp_files=offset_to_tmp_files,
                        offset_to_tmp_dir=offset_to_tmp_dir)
                    cache_ct = 0

            offset_to_cache[current_offset][cache_ct, :, :] = arr
            offset_to_valid_cache[current_offset] = cache_ct+1

    _dump_timeseries_caches(
        offset_to_cache=offset_to_cache,
        offset_to_valid_cache=offset_to_valid_cache,
        offset_to_tmp_files=offset_to_tmp_files,
        offset_to_tmp_dir=offset_to_tmp_dir)

    for offset in offset_to_tmp_files:
        _gather_timeseries_caches(
            file_path_list=offset_to_tmp_files[offset],
            final_output_path=offset_to_path[offset])


def _gather_timeseries_caches(
        file_path_list: List,
        final_output_path: pathlib.Path):

    n_frames = 0
    fov_shape = None
    video_dtype = None
    for file_path in file_path_list:
        with h5py.File(file_path, 'r') as in_file:
            n_frames += in_file['data'].shape[0]
            this_fov_shape = in_file['data'].shape[1:]
            if fov_shape is None:
                fov_shape = this_fov_shape
                video_dtype = in_file['data'].dtype
            else:
                if fov_shape != this_fov_shape:
                    raise RuntimeError(
                        "Inconsistent FOV shape\n"
                        f"{fov_shape}\n{this_fov_shape}")

    chunk_size = n_frames // 100
    if chunk_size < 100:
        chunk_size = n_frames

    with h5py.File(final_output_path, 'w') as out_file:
        out_file.create_dataset(
            'data',
            shape=(n_frames, fov_shape[0], fov_shape[1]),
            dtype=video_dtype,
            chunks=(chunk_size, fov_shape[0], fov_shape[1]))

        i0 = 0
        for file_path in file_path_list:
            with h5py.File(file_path, 'r') as in_file:
                chunk = in_file['data'][()]
                out_file['data'][i0:i0+chunk.shape[0], :, :] = chunk
                i0 += chunk.shape[0]
            os.unlink(file_path)


def _dump_timeseries_caches(
        offset_to_cache: Dict,
        offset_to_valid_cache: Dict,
        offset_to_tmp_files: Dict,
        offset_to_tmp_dir: Dict):

    for offset in offset_to_cache:
        tmp_dir = offset_to_tmp_dir[offset]
        valid = offset_to_valid_cache[offset]
        if valid < 0:
            continue
        cache = offset_to_cache[offset][:valid, :, :]

        tmp_path = tempfile.mkstemp(
                        dir=tmp_dir,
                        suffix='.h5')

        tmp_path = pathlib.Path(tmp_path[1])

        with h5py.File(tmp_path, 'w') as out_file:
            out_file.create_dataset(
                'data', data=cache)
        offset_to_tmp_files[offset].append(tmp_path)
        offset_to_valid_cache[offset] = -1
