from typing import Optional, Dict, List
import tifffile
import h5py
import pathlib
import numpy as np
import datetime
import os
import shutil
import json
import time
import tempfile

from ophys_etl.utils.tempfile_util import (
    mkstemp_clean)


def split_timeseries_tiff(
        tiff_path: pathlib.Path,
        offset_to_path: Dict,
        tmp_dir: Optional[pathlib.Path] = None,
        dump_every: int = 1000,
        logger: Optional[callable] = None,
        metadata: Optional[dict] = None) -> None:
    """
    Split a timeseries TIFF containing multiple mesoscope
    movies into individual HDF5 files.

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the timeseries TIFF file

    offset_to_path: Dict
        A dict mapping the offset corresponding to each timeseries
        to the path to the HDF5 file where that timeseries will be
        written (i.e. offset_to_path[0] = 'fileA.h5',
        offset_to_path[1] = 'fileB.h5', offset_to_path[2] = 'fileC.h5')
        corresponds to a TIFF file whose pages are arranged like
        [fileA, fileB, fileC, fileA, fileB, fileC, fileA, fileB, fileC....]

    tmp_dir: Optional[pathlib.Path]
        Directory where temporary files are written (if None,
        the temporary files corresponding to each HDF5 file will
        be written to the directory where the final HDF5 is meant
        to be written)

    dump_every: int
        Write frames to the temporary files every dump_every
        frames per movie.

    logger: Optional[callable]
        Log statements will be written to logger.info()

    metadata: Optional[dict]
        The metadata read by tifffile.read_scanimage_metadata.
        If not None, will be serialized and stored as a bytestring
        in the HDF5 file.

    Returns
    -------
    None
        Timeseries are written to HDF5 files specified in offset_to_path

    Notes
    -----
    Because there is no way to get the number of pages in a BigTIFF
    without (expensively) scanning through all of the pages, this
    method operates by iterating through the pages once, writing
    the pages corresponding to each timeseries to a series of temporary
    files associated with that timeseries. Once all of the pages have
    been written to the temporary files, the temporary files associated
    with each timeseries are joined into the appropriate HDF5 files
    and deleted.
    """

    # Create a unique temporary directory with an unambiguous
    # name so that if clean-up gets interrupted by some
    # catastrophic failure we will know what the directory
    # was for.
    now = datetime.datetime.now()
    timestamp = (f"{now.year}_{now.month}_"
                 f"{now.day}_{now.hour}_"
                 f"{now.minute}_{now.second}")

    tmp_prefix = f"mesoscope_timeseries_tmp_{timestamp}_"
    directories_to_clean = []

    if tmp_dir is not None:
        actual_tmp_dir = pathlib.Path(
                tempfile.mkdtemp(
                    dir=tmp_dir,
                    prefix=tmp_prefix))
        directories_to_clean.append(actual_tmp_dir)

    offset_to_tmp_files = dict()
    offset_to_tmp_dir = dict()
    for offset in offset_to_path:
        if tmp_dir is not None:
            offset_to_tmp_dir[offset] = actual_tmp_dir
        else:
            pth = offset_to_path[offset]
            actual_tmp_dir = pathlib.Path(
                    tempfile.mkdtemp(
                        dir=pth.parent,
                        prefix=tmp_prefix))
            offset_to_tmp_dir[offset] = actual_tmp_dir
            directories_to_clean.append(actual_tmp_dir)
        offset_to_tmp_files[offset] = []

    try:
        _split_timeseries_tiff(
            tiff_path=tiff_path,
            offset_to_path=offset_to_path,
            offset_to_tmp_files=offset_to_tmp_files,
            offset_to_tmp_dir=offset_to_tmp_dir,
            dump_every=dump_every,
            logger=logger,
            metadata=metadata)
    finally:
        for offset in offset_to_tmp_files:
            for tmp_pth in offset_to_tmp_files[offset]:
                if tmp_pth.exists():
                    os.unlink(tmp_pth)

        # clean up temporary directories
        for dir_pth in directories_to_clean:
            if dir_pth.exists():
                shutil.rmtree(dir_pth)


def _split_timeseries_tiff(
        tiff_path: pathlib.Path,
        offset_to_path: Dict,
        offset_to_tmp_files: Dict,
        offset_to_tmp_dir: Dict,
        dump_every: int = 1000,
        logger: Optional[callable] = None,
        metadata: Optional[dict] = None) -> None:
    """
    Method to do the work behind split_timeseries_tiff

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the timeseries TIFF being split

    offset_to_path: Dict
        Dict mapping offset to final HDF5 path (see
        split_timeseries_tiff for explanation)

    offset_to_tmp_files: Dict
        An empty dict for storing the lists of temporary
        files generated for each individual timeseries.
        This method wil populate the dict in place.

    offset_to_tmp_dir: Dict
        A dict mapping offset to the directory where
        the corresponding temporary files will be written

    dump_every: int
        Write to the temporary files every dump_every frames

    logger: Optional[callable]
        Log statements will be written to logger.info()

    metadata: Optional[dict]
        The metadata read by tifffile.read_scanimage_metadata.
        If not None, will be serialized and stored as a bytestring
        in the HDF5 file.

    Returns
    -------
    None
        Timeseries data is written to the paths specified in
        offset_to_path
    """

    if logger is not None:
        logger.info(f"Splitting {tiff_path}")

    max_offset = max(list(offset_to_path.keys()))

    fov_shape = None
    video_dtype = None
    offset_to_cache = dict()
    offset_to_valid_cache = dict()

    t0 = time.time()
    page_ct = 0
    with tifffile.TiffFile(tiff_path, mode='rb') as tiff_file:
        current_offset = -1
        cache_ct = 0
        for page in tiff_file.pages:
            page_ct += 1
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
                    if logger is not None:
                        duration = time.time()-t0
                        msg = f"Iterated through {page_ct} TIFF pages "
                        msg += f"in {duration:.2e} seconds"
                        logger.info(msg)

            offset_to_cache[current_offset][cache_ct, :, :] = arr
            offset_to_valid_cache[current_offset] = cache_ct+1

    _dump_timeseries_caches(
        offset_to_cache=offset_to_cache,
        offset_to_valid_cache=offset_to_valid_cache,
        offset_to_tmp_files=offset_to_tmp_files,
        offset_to_tmp_dir=offset_to_tmp_dir)

    if logger is not None:
        duration = time.time()-t0
        msg = f"Iterated through all {page_ct} TIFF pages "
        msg += f"in {duration:.2e} seconds"
        logger.info(msg)

    for offset in offset_to_tmp_files:
        _gather_timeseries_caches(
            file_path_list=offset_to_tmp_files[offset],
            final_output_path=offset_to_path[offset],
            metadata=metadata)
        if logger is not None:
            duration = time.time()-t0
            msg = f"Wrote {offset_to_path[offset]} after "
            msg += f"{duration:.2e} seconds"
            logger.info(msg)

    if logger is not None:
        duration = time.time()-t0
        msg = f"Split {tiff_path} in {duration:.2e} seconds"
        logger.info(msg)


def _gather_timeseries_caches(
        file_path_list: List[pathlib.Path],
        final_output_path: pathlib.Path,
        metadata: Optional[dict] = None) -> None:
    """
    Take a list of HDF5 files containing an array 'data' and
    join them into a single HDF5 file with an array 'data' that
    is the result of calling np.stack() on the smaller arrays.

    Parameters
    ----------
    file_path_list: List[pathlib.Path]
        List of paths to files to be joined

    final_output_path: pathlib.Path
        Path to the HDF5 file that is produced by joining
        file_path_list

    metadata: Optional[dict]
        The metadata read by tifffile.read_scanimage_metadata.
        If not None, will be serialized and stored as a bytestring
        in the HDF5 file.

    Return
    ------
    None
        Contents of files in file_path_list are joined into
        final_output_path

    Notes
    -----
    Files in file_path_list are deleted with os.unlink
    after they are joined.
    """
    n_frames = 0
    fov_shape = None
    video_dtype = None
    one_frame = None  # for calculating frame size in memory

    for file_path in file_path_list:
        with h5py.File(file_path, 'r') as in_file:
            n_frames += in_file['data'].shape[0]
            this_fov_shape = in_file['data'].shape[1:]

            if one_frame is None:
                one_frame = in_file['data'][0, :, :]

            if fov_shape is None:
                fov_shape = this_fov_shape
                video_dtype = in_file['data'].dtype
            else:
                if fov_shape != this_fov_shape:
                    raise RuntimeError(
                        "Inconsistent FOV shape\n"
                        f"{fov_shape}\n{this_fov_shape}")

    # apparently, HDF5 chunks sizes must be less than 4 GB;
    # figure out how many frames fit in 3GB (just in case)
    # and set that as the maximum chunk size for the final
    # HDF5 file.
    three_gb = 3*1024**3
    bytes_per_frame = len(one_frame.tobytes())
    max_chunk_size = np.floor(three_gb/bytes_per_frame).astype(int)

    chunk_size = n_frames // 100

    if chunk_size < n_frames:
        chunk_size = n_frames

    if chunk_size > max_chunk_size:
        chunk_size = max_chunk_size

    with h5py.File(final_output_path, 'w') as out_file:

        if metadata is not None:
            serialized_metadata = json.dumps(metadata).encode('utf-8')
            out_file.create_dataset(
                'scanimage_metadata',
                data=serialized_metadata)

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
        offset_to_tmp_dir: Dict) -> None:
    """
    Write cached arrays to temporary files.

    Parameters
    ----------
    offset_to_cache: Dict
        Maps offset (see split_timeseries_tiff) to numpy
        arrays that are being dumped to temporary files.

    offset_to_valid_cache: Dict
        Maps offset to the index in offset_to_cache[offset] that
        is the last valid row (in case the cache was incompletely
        populated), i.e.
        offset_to_cache[offset][:offset_to_valid_cache[offset], :, :]
        is written to the temporary files.

        After this method is run, all entries in offset_to_cache
        are set to -1.

    offset_to_tmp_files: Dict
        Maps offset to list of temporary files that are being written.
        This dict starts out mapping to empty lists. This method
        creates temporary files and populates this dict with paths
        to them.

    offset_to_tmp_dir: Dict
       Maps offset to directory where temporary files will be written

    Returns
    -------
    None
        This metod writes the data input through offset_to_cache
        to temporary files (that this method creates and logs in
        offset_to_tmp_files)
    """

    for offset in offset_to_cache:
        tmp_dir = offset_to_tmp_dir[offset]
        valid = offset_to_valid_cache[offset]
        if valid < 0:
            continue
        cache = offset_to_cache[offset][:valid, :, :]

        tmp_path = mkstemp_clean(
                        dir=tmp_dir,
                        suffix='.h5')

        tmp_path = pathlib.Path(tmp_path)

        # append path first so the code knows to clean up
        # the file in case file-creation gets interrupted
        offset_to_tmp_files[offset].append(tmp_path)

        with h5py.File(tmp_path, 'w') as out_file:
            out_file.create_dataset(
                'data', data=cache)
        offset_to_valid_cache[offset] = -1
