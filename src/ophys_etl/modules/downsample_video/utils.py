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
        kernel_size: Optional[int],
        input_slice: Tuple[int, int],
        chunk_validity: dict,
        output_lock: multiprocessing.managers.AcquirerProxy) -> None:
    """
    A worker that can be called by multiprocessing to read a chunk of the
    input video from an HDF5 file, downsample and median filter it, and
    write it to an output HDF5 file (presumably in scratch space).

    Parameters
    ----------
    input_path: pathlib.Path
        Path to the input movie

    input_hz: float
        Frame rate of the input movie in Hz

    output_path: pathlib.Path
         Path to the HDF5 file where the processed movie will be stored

    output_hz: float
        Frame rate of the output movie in Hz (in case it is downsampled
        relative to the input movie)

    kernel_size: Optional[int]
        The size of the median filter kernel. This filter is applied to
        the frames of the output movie after downsampling. If None,
        no median filter is applied.

    input_slice: Tuple[int, int]
        The first (inclusive) and last (exclusive) frame of the
        input movie to be processed by this worker.

    chunk_validity: dict
        A multiprocessing manager dict to map input_slice[0] to
        a boolean indicating whether or not the chunk was successfully
        written and a message explaining why.

    output_lock: multiprocessing.managers.AcquirerProxy
        A multiprocessing lock to prevent multiple processes from
        trying to write to the output file at the same time.

    Returns
    -------
    None
        Output is written to the HDF5 file at output_path
    """

    t0 = time.time()
    frames_to_group = n_frames_from_hz(
            input_hz,
            output_hz)

    if input_slice[0] % frames_to_group != 0:
        msg = "input_slice[0] must be an integer multiple of "
        msg += "n_frame_from_hz(input_hz, output_hz)\n"
        msg += f"input_slice[0]: {input_slice[0]}\n"
        msg += f"n_frames_from_hz: {frames_to_group}\n"
        chunk_validity[input_slice[0]] = (False, msg)
        raise RuntimeError(msg)

    with h5py.File(input_path, 'r') as in_file:
        video_data = in_file['data'][input_slice[0]:input_slice[1], :, :]

    if frames_to_group > 1:
        video_data = downsample_array(video_data,
                                      input_fps=input_hz,
                                      output_fps=output_hz,
                                      strategy='average')

    if kernel_size is not None and kernel_size > 0:
        video_data = apply_median_filter_to_video(video_data,
                                                  kernel_size)
    start_index = input_slice[0] // frames_to_group
    end_index = start_index + video_data.shape[0]
    with output_lock:
        with h5py.File(output_path, 'a') as out_file:
            out_file['data'][start_index:end_index, :, :] = video_data
        duration = time.time()-t0
        print(f'completed chunk in {duration:.2e} seconds')
        chunk_validity[input_slice[0]] = (True, '')


def create_downsampled_video_h5(
        input_path: pathlib.Path,
        input_hz: float,
        output_path: pathlib.Path,
        output_hz: float,
        kernel_size: Optional[int],
        n_processors: int) -> None:
    """
    Use multiprocessing to read a movie from an HDF5 file, downsample
    and median filter it, and write it to another HDF5 file (presumably
    in scratch)

    Parameters
    ----------
    input_path: pathlib.Path
        Path to the input movie

    input_hz:
        frame rate of the input movie in Hz

    output_path: pathlib.Path
        Path to the HDF5 file where the processed movie will be written

    output_hz:
        frame rate of the output movie in Hz (in case it is downsampled)

    kernel_size: Optional[int]
        The size of the median filter kernel. This filter is applied to
        the frames of the output movie after downsampling. If None,
        no median filter is applied.

    n_processors: int
        Number of parallel worker processes to invoke when processing the
        movie.

    Returns
    -------
    None
        Output is written to the HDF5 file specified at output_path
    """

    with h5py.File(input_path, 'r') as in_file:
        input_video_shape = in_file['data'].shape

    # determine how many frames are going to be grouped together
    # by downsampling
    frames_to_group = n_frames_from_hz(
                            input_hz,
                            output_hz)

    # determine how many frames to pass to each parallel process
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
    validity_dict = mgr.dict()
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
                      validity_dict,
                      output_lock))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    msg = ''
    for k in validity_dict:
        if validity_dict[k][0]:
            continue
        msg += '\n'
        msg += validity_dict[k][1]
    if len(msg) > 0:
        raise RuntimeError(msg)


def _write_array_to_video(
        video_path: pathlib.Path,
        video_array: np.ndarray,
        fps: int,
        quality: int):
    """
    Write a video array to a video file (mp4, avi, etc.) using
    imageio.

    Parameters
    ----------
    video_path: pathlib.Path
        Path to the putput file

    video_array: np.ndarray
        Numpy array containing video data. Probably of shape
        (ntime, nrows, ncols, 3)

    fps: int
        Frames per second to write with imageio

    quality: int
        An integer from 0-9 (inclusive) denoting the quality of the
        video to be written

    Returns
    -------
    None
        Output is written to the specified file path
    """

    if video_path.name.endswith('avi'):
        pixelformat = 'yuvj420p'
        codec = 'mjpeg'
    else:
        pixelformat = 'yuv420p'
        codec = 'libx264'

    imageio.mimsave(video_path,
                    video_array,
                    fps=fps,
                    quality=quality,
                    pixelformat=pixelformat,
                    codec=codec)

    print(f'wrote {video_path}')


def _min_max_from_h5(
        h5_path: pathlib.Path,
        quantiles: Optional[Tuple[float, float]],
        border: int = 50) -> Tuple[float, float]:
    """
    Get the normalizing minimum and maximum pixel values from
    a movie, ignoring pixels at the border.

    Parameters
    ----------
    h5_path: pathlib.Path
        Path to the movie

    quantiles: Optional[Tuple[float, float]]
        Quantiles to use for normalization (if None, get
        minimum and maximum values)

    border: int
        Number of pixels to ignore at the border of the field of view.
        Note: if border would exclude all pixels in the frame, border is
        set to zero.

    Returns
    -------
    norm_min, norm_max: Tuple[float, float]
        The minimum and maximum (or specified quantiles)
        of the pixel values from the movie.
    """

    with h5py.File(h5_path, 'r') as in_file:
        video_shape = in_file['data'].shape

        if 2*border > video_shape[1] or 2*border > video_shape[2]:
            border = 0

        # clip motion border, just in case
        full_data = in_file['data'][:,
                                    border:video_shape[1]-border,
                                    border:video_shape[2]-border]
        if quantiles is not None:
            q0, q1 = np.quantile(full_data, quantiles)
        else:
            q0 = full_data.min()
            q1 = full_data.max()
        print('got normalization')

    return q0, q1


def _video_array_from_h5(
        h5_path: pathlib.Path,
        min_val: float,
        max_val: float,
        reticle: bool = True,
        d_reticle: int = 64) -> np.ndarray:
    """
    Read in an HDF5 file and convert it into a numpy array that
    can be passed to _write_array_to_video

    Parameters
    ----------
    h5_path: pathlib.Path
        Path to the HDF5 file containing the video data

    min_val: float
        The minimum value at which to clip the video

    max_val: float
        The maximum value at which to clip the video

    reticle: bool
        If True, add a grid of red lines to the video, to help guide
        the eye

    d_reticle: int
        Spacing between reticles

    Returns
    -------
    video_as_uint: np.ndarray
        A (ntime, nrows, ncols, 3) array of uints representing the
        RGB video.
    """

    with h5py.File(h5_path, 'r') as in_file:
        video_shape = in_file['data'].shape

        video_as_uint = np.zeros((video_shape[0],
                                  video_shape[1],
                                  video_shape[2],
                                  3), dtype=np.uint8)
        dt = 500
        for i0 in range(0, video_shape[0], dt):
            i1 = i0+dt
            data = in_file['data'][i0:i1, :, :].astype(float)
            data = np.where(data > min_val, data, min_val)
            data = np.where(data < max_val, data, max_val)
            delta = max_val-min_val
            data = np.round(255.0*(data-min_val)/delta).astype(np.uint8)
            for ic in range(3):
                video_as_uint[i0:i1, :, :, ic] = data

    print('constructed video_as_uint')

    if reticle:
        for ii in range(d_reticle, video_shape[1], d_reticle):
            old_vals = np.copy(video_as_uint[:, ii:ii+2, :, :])
            new_vals = np.zeros(old_vals.shape, dtype=np.uint8)
            new_vals[:, :, :, 0] = 255
            new_vals = (new_vals//2) + (old_vals//2)
            new_vals = new_vals.astype(np.uint8)
            video_as_uint[:, ii:ii+2, :, :] = new_vals
        for ii in range(d_reticle, video_shape[2], d_reticle):
            old_vals = np.copy(video_as_uint[:, :, ii:ii+2, :])
            new_vals = np.zeros(old_vals.shape, dtype=np.uint8)
            new_vals[:, :, :, 0] = 255
            new_vals = (new_vals//2) + (old_vals//2)
            new_vals = new_vals.astype(np.uint8)
            video_as_uint[:, :, ii:ii+2, :] = new_vals

    print('added reticles')

    return video_as_uint


def create_downsampled_video(
        input_path: pathlib.Path,
        input_hz: float,
        video_path: pathlib.Path,
        output_hz: float,
        kernel_size: Optional[int],
        n_processors: int,
        quality: int = 5,
        quantiles: Tuple[float, float] = (0.1, 0.99),
        reticle: bool = True,
        speed_up_factor: int = 8,
        tmp_dir: Optional[pathlib.Path] = None) -> None:
    """
    Create a video file (mp4, avi, etc.) from an HDF5 file, applying
    downsampling and a median filter if desired.

    Parameters
    ----------
    input_path: pathlib.Path
        Path to the HDF5 file containing the movie data

    input_hz:
        Frame rate of the input movie in Hz

    video_path: Pathlib.path
        Path to the video file to be written

    output_hz: float
        Frame rate of the output movie in Hz (set lower than input_hz
        if you want to apply downsampling to the movie)

    kernel_size: Optional[int]
        Size of the median filter kernel to be applied to the downsampled
        movie (if None, no median filter will be applied)

    n_processors: int
        Number of parallel processes to be used when processing the movie

    quality: int
        A value between 0-9 (inclusive) denoting the quality of the movie
        to be written (higher number means better quality)

    quantiles: Tuple[float, float]
        The quantiles to which to clip the movie before writing it to video

    reticle: bool
        If True, add a grid of red lines to the movie to guide the eye

    speed_up_factor: int
        Factor by which to speed up the movie *after downsampling* when writing
        to video (in case you want a smaller file that can be played back
        faster)

    tmp_dir: Optional[pathlib.Path]
        Scratch directory to use during processing. When applying the median
        filter, the code writes the filtered movie to disk, rather than try
        to keep two copies of the movie in memory. This gives the user the
        option to specify where the scratch copy of the movie is written.
        If None, the scratch movie will be written to the system's default
        scratch space.

    Returns
    -------
    None
        Output is written to the specified movie file
    """

    with tempfile.TemporaryDirectory(dir=tmp_dir) as this_tmp_dir:
        tmp_h5 = tempfile.mkstemp(dir=this_tmp_dir, suffix='.h5')[1]
        tmp_h5 = pathlib.Path(tmp_h5)
        print(f'writing h5py to {tmp_h5}')

        create_downsampled_video_h5(
            input_path, input_hz,
            tmp_h5, output_hz,
            kernel_size,
            n_processors)

        print(f'wrote temp h5py to {tmp_h5}')

        (min_val,
         max_val) = _min_max_from_h5(tmp_h5, quantiles)

        video_array = _video_array_from_h5(
                tmp_h5,
                min_val,
                max_val,
                reticle)

        tmp_h5.unlink()

        _write_array_to_video(
            video_path,
            video_array,
            int(speed_up_factor*output_hz),
            quality)


def create_side_by_side_video(
        video_0_path: pathlib.Path,
        video_1_path: pathlib.Path,
        input_hz: float,
        output_path: pathlib.Path,
        output_hz: float,
        kernel_size: Optional[int],
        n_processors: int,
        quality: int = 5,
        quantiles: Tuple[float, float] = (0.1, 0.99),
        reticle: bool = True,
        speed_up_factor: int = 8,
        tmp_dir: Optional[pathlib.Path] = None):
    """
    Create a video file (mp4, avi, etc.) from two HDF5 files, showing the
    movies side by side for easy comparison, applying downsampling and a
    median filter if desired.

    Parameters
    ----------
    video_0_path: pathlib.Path
        Path to the HDF5 file containing the movie to be shown on the left

    video_1_path: pathlib.Path
        Path to the HDF5 file containing the movie to be shown on the right

    input_hz:
        Frame rate of the input movie in Hz (assume it is the same for
        video_0 and video_1, since they are presumably the same movie
        in different states of motion correction)

    video_path: Pathlib.path
        Path to the video file to be written

    output_hz: float
        Frame rate of the output movie in Hz (set lower than input_hz
        if you want to apply downsampling to the movie)

    kernel_size: Optional[int]
        Size of the median filter kernel to be applied to the downsampled
        movie (if None, no median filter will be applied)

    n_processors: int
        Number of parallel processes to be used when processing the movie

    quality: int
        A value between 0-9 (inclusive) denoting the quality of the movie
        to be written (higher number means better quality)

    quantiles: Tuple[float, float]
        The quantiles to which to clip the movie before writing it to video

    reticle: bool
        If True, add a grid of red lines to the movie to guide the eye

    speed_up_factor: int
        Factor by which to speed up the movie *after downsampling* when writing
        to video (in case you want a smaller file that can be played back
        faster)

    tmp_dir: Optional[pathlib.Path]
        Scratch directory to use during processing. When applying the median
        filter, the code writes the filtered movie to disk, rather than try
        to keep two copies of the movie in memory. This gives the user the
        option to specify where the scratch copy of the movie is written.
        If None, the scratch movie will be written to the system's default
        scratch space.

    Returns
    -------
    None
        Output is written to the specified movie file
    """

    with h5py.File(video_0_path, 'r') as in_file:
        video_0_shape = in_file['data'].shape
    with h5py.File(video_1_path, 'r') as in_file:
        video_1_shape = in_file['data'].shape

    if video_0_shape != video_1_shape:
        msg = 'Videos need to be the same shape\n'
        msg += f'{video_0_path}: {video_0_shape}\n'
        msg += f'{video_1_path}: {video_1_shape}'
        raise RuntimeError(msg)

    # number of pixels in a blank column between the movies
    gap = 16

    with tempfile.TemporaryDirectory(dir=tmp_dir) as this_tmp_dir:

        tmp_0_h5 = tempfile.mkstemp(dir=this_tmp_dir, suffix='.h5')[1]
        tmp_0_h5 = pathlib.Path(tmp_0_h5)

        tmp_1_h5 = tempfile.mkstemp(dir=this_tmp_dir, suffix='.h5')[1]
        tmp_1_h5 = pathlib.Path(tmp_1_h5)

        create_downsampled_video_h5(
            video_0_path, input_hz,
            tmp_0_h5, output_hz,
            kernel_size,
            n_processors)

        (min_0,
         max_0) = _min_max_from_h5(tmp_0_h5, quantiles)

        print(f'wrote {video_0_path} to {tmp_0_h5}')

        create_downsampled_video_h5(
            video_1_path, input_hz,
            tmp_1_h5, output_hz,
            kernel_size,
            n_processors)

        (min_1,
         max_1) = _min_max_from_h5(tmp_1_h5, quantiles)

        print(f'wrote {video_1_path} to {tmp_1_h5}')

        video_array = np.zeros((video_0_shape[0],
                                video_0_shape[1],
                                gap+2*video_0_shape[2],
                                3), dtype=np.uint8)

        video_0_uint = _video_array_from_h5(tmp_0_h5,
                                            min_0,
                                            max_0,
                                            reticle)

        tmp_0_h5.unlink()

        video_array = np.zeros((video_0_uint.shape[0],
                                video_0_shape[1],
                                gap+2*video_0_shape[2],
                                3), dtype=np.uint8)

        video_array[:, :,
                    :video_0_shape[2], :] = video_0_uint

        del video_0_uint

        video_array[:, :,
                    video_0_shape[2]:video_0_shape[2]+gap, :] = 125

        video_array[:, :,
                    video_0_shape[2]+gap:, :] = _video_array_from_h5(
                                                      tmp_1_h5,
                                                      min_1,
                                                      max_1,
                                                      reticle)

        tmp_1_h5.unlink()

        print('created video array')

        _write_array_to_video(
            output_path,
            video_array,
            int(speed_up_factor*output_hz),
            quality)
