from typing import Tuple, Optional, Callable
import pathlib
import h5py
import numpy as np
import skimage.measure as skimage_measure

from ophys_etl.utils.array_utils import (
    downsample_array,
    n_frames_from_hz)

import multiprocessing
import multiprocessing.managers
import time

import imageio
import tempfile

import logging


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logger.setLevel(logging.INFO)


def create_downsampled_video(
        input_path: pathlib.Path,
        input_hz: float,
        output_path: pathlib.Path,
        output_hz: float,
        spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]],
        n_processors: int,
        quality: int = 5,
        quantiles: Tuple[float, float] = (0.1, 0.99),
        reticle: bool = True,
        speed_up_factor: int = 8,
        tmp_dir: Optional[pathlib.Path] = None,
        video_dtype: type = np.uint8) -> None:
    """
    Create a video file (mp4, avi, etc.) from an HDF5 file, applying
    downsampling and a median filter if desired.

    Parameters
    ----------
    input_path: pathlib.Path
        Path to the HDF5 file containing the movie data

    input_hz:
        Frame rate of the input movie in Hz

    output_path: Pathlib.path
        Path to the video file to be written

    output_hz: float
        Frame rate of the output movie in Hz (set lower than input_hz
        if you want to apply downsampling to the movie)

    spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]]
        The function (if any) used to spatially filter frames after
        downsampling. Accepts an np.ndarray that is the input video;
        returns an np.ndarray that is the spatially filtered video.

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
        to video (in case you want a file that plays back faster)

    tmp_dir: Optional[pathlib.Path]
        Scratch directory to use during processing. When applying the median
        filter, the code writes the filtered movie to disk, rather than try
        to keep two copies of the movie in memory. This gives the user the
        option to specify where the scratch copy of the movie is written.
        If None, the scratch movie will be written to the system's default
        scratch space.

    video_dtype: type
        Type to which the video will be cast (must be either
        np.uint8 or np.uint16)

    Returns
    -------
    None
        Output is written to the specified movie file
    """

    video_array = _downsampled_video_array_from_h5(
                        input_path=input_path,
                        input_hz=input_hz,
                        output_hz=output_hz,
                        spatial_filter=spatial_filter,
                        n_processors=n_processors,
                        quantiles=quantiles,
                        reticle=reticle,
                        tmp_dir=tmp_dir,
                        video_dtype=video_dtype)

    _write_array_to_video(
        output_path,
        video_array,
        int(speed_up_factor*output_hz),
        quality)


def create_side_by_side_video(
        left_video_path: pathlib.Path,
        right_video_path: pathlib.Path,
        input_hz: float,
        output_path: pathlib.Path,
        output_hz: float,
        spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]],
        n_processors: int,
        quality: int = 5,
        quantiles: Tuple[float, float] = (0.1, 0.99),
        reticle: bool = True,
        speed_up_factor: int = 8,
        tmp_dir: Optional[pathlib.Path] = None,
        video_dtype: type = np.uint8):
    """
    Create a video file (mp4, avi, etc.) from two HDF5 files, showing the
    movies side by side for easy comparison, applying downsampling and a
    median filter if desired.

    Parameters
    ----------
    left_video_path: pathlib.Path
        Path to the HDF5 file containing the movie to be shown on the left

    right_video_path: pathlib.Path
        Path to the HDF5 file containing the movie to be shown on the right

    input_hz:
        Frame rate of the input movie in Hz (assume it is the same for
        right_video and left_video, since they are presumably the same
        movie in different states of motion correction)

    output_path: Pathlib.path
        Path to the video file to be written

    output_hz: float
        Frame rate of the output movie in Hz (set lower than input_hz
        if you want to apply downsampling to the movie)

    spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]]
        The function (if any) used to spatially filter frames after
        downsampling. Accepts an np.ndarray that is the input video;
        returns an np.ndarray that is the spatially filtered video.

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

    video_dtype: type
        Type to which the video will be cast (must be either
        np.uint8 or np.uint16)

    Returns
    -------
    None
        Output is written to the specified movie file
    """

    with h5py.File(left_video_path, 'r') as in_file:
        left_shape = in_file['data'].shape
    with h5py.File(right_video_path, 'r') as in_file:
        right_shape = in_file['data'].shape

    if left_shape != right_shape:
        msg = 'Videos need to be the same shape\n'
        msg += f'{left_video_path}: {left_shape}\n'
        msg += f'{right_video_path}: {right_shape}'
        raise RuntimeError(msg)

    # so we do not use them again; output videos
    # need to be the same shape as the smoothed
    # uint arrays
    del left_shape
    del right_shape

    # number of pixels in a blank column between the movies
    gap = 16

    left_uint = _downsampled_video_array_from_h5(
                        input_path=left_video_path,
                        input_hz=input_hz,
                        output_hz=output_hz,
                        spatial_filter=spatial_filter,
                        n_processors=n_processors,
                        quantiles=quantiles,
                        reticle=reticle,
                        tmp_dir=tmp_dir,
                        video_dtype=video_dtype)

    video_array = np.zeros((left_uint.shape[0],
                            left_uint.shape[1],
                            gap+2*left_uint.shape[2],
                            3), dtype=video_dtype)

    video_array[:, :,
                :left_uint.shape[2], :] = left_uint

    single_video_ncols = left_uint.shape[2]

    del left_uint

    half_val = int(np.iinfo(video_dtype).max//2)

    # make the gap between videos gray
    video_array[:,
                :,
                single_video_ncols:single_video_ncols+gap,
                :] = half_val

    video_array[:, :,
                single_video_ncols+gap:,
                :] = _downsampled_video_array_from_h5(
                                             input_path=right_video_path,
                                             input_hz=input_hz,
                                             output_hz=output_hz,
                                             spatial_filter=spatial_filter,
                                             n_processors=n_processors,
                                             quantiles=quantiles,
                                             reticle=reticle,
                                             tmp_dir=tmp_dir,
                                             video_dtype=video_dtype)

    logger.info('created video array')

    _write_array_to_video(
        output_path,
        video_array,
        int(speed_up_factor*output_hz),
        quality)


def _video_worker(
        input_path: pathlib.Path,
        input_hz: float,
        output_path: pathlib.Path,
        output_hz: float,
        spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]],
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

    spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]]
        The function (if any) used to spatially filter frames after
        downsampling. Accepts an np.ndarray that is the input video;
        returns an np.ndarray that is the spatially filtered video.

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

    if spatial_filter is not None:
        video_data = spatial_filter(video_data)

    start_index = input_slice[0] // frames_to_group
    end_index = start_index + video_data.shape[0]
    with output_lock:
        with h5py.File(output_path, 'a') as out_file:
            out_file['data'][start_index:end_index, :, :] = video_data
        duration = time.time()-t0
        logger.info(f'completed chunk in {duration:.2e} seconds')
        chunk_validity[input_slice[0]] = (True, '')


def create_downsampled_video_h5(
        input_path: pathlib.Path,
        input_hz: float,
        output_path: pathlib.Path,
        output_hz: float,
        spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]],
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

    spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]]
        The function (if any) used to spatially filter frames after
        downsampling. Accepts an np.ndarray that is the input video;
        returns an np.ndarray that is the spatially filtered video.

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
        frame_size = _get_post_filter_frame_size(
                           example_video=in_file['data'][:2, :, :],
                           spatial_filter=spatial_filter)

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
                                       frame_size[0],
                                       frame_size[1]),
                                chunks=(max(1, n_frames_out//100),
                                        frame_size[0],
                                        frame_size[1]),
                                dtype=float)

    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()
    validity_dict = mgr.dict()
    process_list = []

    input_chunks = []
    for i0 in range(0, input_video_shape[0], n_frames_per_chunk):
        logger.info(f'starting {i0} -> {input_video_shape[0]}')
        input_chunks.append(i0)
        p = multiprocessing.Process(
                target=_video_worker,
                args=(input_path,
                      input_hz,
                      output_path,
                      output_hz,
                      spatial_filter,
                      (i0, i0+n_frames_per_chunk),
                      validity_dict,
                      output_lock))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    msg = ''
    for k in input_chunks:
        if k not in validity_dict:
            msg += f'\nchunk {k} was not completed'
            continue

        if validity_dict[k][0]:
            continue
        msg += '\n'
        msg += validity_dict[k][1]
    if len(msg) > 0:
        raise RuntimeError(msg)


def _get_post_filter_frame_size(
        example_video: np.ndarray,
        spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]]
        ) -> Tuple[int, int]:
    """
    Parameters
    ----------
    example_video: np.ndarray
        (ntime, nrows, ncols) input video data

    spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]
        spatial filter to be applied to video, transforming it
        from (ntime, nrows0, ncols0) -> (ntime, nrows1, ncols1)

    Returns
    -------
    frame_shape: Tuple[int, int]
        (nrows1, ncols1) -- spatial shape of frame after spatial_filter
    """
    if spatial_filter is not None:
        filtered_video = spatial_filter(example_video)
    else:
        filtered_video = example_video

    return filtered_video.shape[1:]


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
        Path to the output file

    video_array: np.ndarray
        Numpy array containing video data. Probably of shape
        (ntime, nrows, ncols, 3)

    fps: int
        Frames per second to write with imageio (ignored if writing TIFF)

    quality: int
        An integer from 0-9 (inclusive) denoting the quality of the
        video to be written (ignored if writing TIFF)

    Returns
    -------
    None
        Output is written to the specified file path.

    Notes
    -----
    If writing a TIFF, data will be saved in grayscale, since TIFFs
    are limited in size to 4 GB and our videos typically exceed this
    size if all three colors channels are saved (this really only
    impacts the reticles, which are red by default, but will be
    white in the TIFFs).
    """

    logger.info(f"writing array of shape {video_array.shape}")

    if video_path.suffix in ('.tiff', '.tif'):
        if len(video_array.shape) == 4:
            imageio.mimsave(video_path,
                            video_array[:, :, :, 0])
        else:
            imageio.mimsave(video_path, video_array)
    else:
        if video_path.suffix == '.avi':
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

    logger.info(f'wrote {video_path}')


def _min_max_from_h5(
        h5_path: pathlib.Path,
        quantiles: [Tuple[float, float]] = (0.0, 1.0),
        border: int = 50) -> Tuple[float, float]:
    """
    Get the normalizing minimum and maximum pixel values from
    a movie, ignoring pixels at the border.

    Parameters
    ----------
    h5_path: pathlib.Path
        Path to the movie

    quantiles: Tuple[float, float]
        Quantiles to use for normalization

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
        q0, q1 = np.quantile(full_data, quantiles)
        logger.info('got normalization')

    return q0, q1


def _video_array_from_h5(
        h5_path: pathlib.Path,
        min_val: float = -np.inf,
        max_val: float = np.inf,
        reticle: bool = True,
        d_reticle: int = 64,
        video_dtype: type = np.uint8) -> np.ndarray:
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

    video_dtype: type
        Type to which the video will be cast (must be either
        np.uint8 or np.uint16)

    Returns
    -------
    video_as_uint: np.ndarray
        A (ntime, nrows, ncols, 3) array of uints representing the
        RGB video.
    """

    if video_dtype not in (np.uint8, np.uint16):
        msg = f'video_dtype: {video_dtype}\n'
        msg += 'is not legal; must be either np.uint8 or np.uint16'
        raise ValueError(msg)

    max_cast_value = int(np.iinfo(video_dtype).max)

    with h5py.File(h5_path, 'r') as in_file:
        video_shape = in_file['data'].shape

        video_as_uint = np.zeros((video_shape[0],
                                  video_shape[1],
                                  video_shape[2],
                                  3),
                                 dtype=video_dtype)
        dt = 500
        for i0 in range(0, video_shape[0], dt):
            i1 = i0+dt
            data = in_file['data'][i0:i1, :, :].astype(float)
            data = np.clip(data, min_val, max_val)
            delta = max_val-min_val
            data -= min_val
            data = np.round(max_cast_value*data/delta).astype(video_dtype)
            for ic in range(3):
                video_as_uint[i0:i1, :, :, ic] = data

    logger.info('constructed video_as_uint')

    if reticle:
        video_as_uint = add_reticle(
                            video_array=video_as_uint,
                            d_reticle=d_reticle)

        logger.info('added reticles')

    return video_as_uint


def add_reticle(
        video_array: np.ndarray,
        d_reticle: int) -> np.ndarray:
    """
    Add reticles to video array

    Parameters
    ----------
    video_array: np.ndarray
        (ntime, nrows, ncols, 3) an RGB representation of a video

    d_reticle: int
        The number of pixels between grid lines

    Returns
    -------
    video_array: np.ndarray
        The same video array with a grid of red lines of the specified
        spacing. The gridline will be applied with alpha=0.25.

    Note
    -----
    Alters video_array in place
    """
    video_dtype = video_array.dtype
    max_cast_value = int(np.iinfo(video_dtype).max)
    video_shape = video_array.shape

    for ii in range(d_reticle, video_shape[1], d_reticle):
        old_vals = np.copy(video_array[:, ii:ii+2, :, :])
        new_vals = np.zeros(old_vals.shape, dtype=video_dtype)
        new_vals[:, :, :, 0] = max_cast_value
        new_vals = (new_vals//4) + (3*(old_vals//4))
        new_vals = new_vals.astype(video_dtype)
        video_array[:, ii:ii+2, :, :] = new_vals
    for ii in range(d_reticle, video_shape[2], d_reticle):
        old_vals = np.copy(video_array[:, :, ii:ii+2, :])
        new_vals = np.zeros(old_vals.shape, dtype=video_dtype)
        new_vals[:, :, :, 0] = max_cast_value
        new_vals = (new_vals//4) + (3*(old_vals//4))
        new_vals = new_vals.astype(video_dtype)
        video_array[:, :, ii:ii+2, :] = new_vals
    return video_array


def _downsampled_video_array_from_h5(
        input_path: pathlib.Path,
        input_hz: float,
        output_hz: float,
        spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]],
        n_processors: int,
        quantiles: Tuple[float, float] = (0.1, 0.99),
        reticle: bool = True,
        tmp_dir: Optional[pathlib.Path] = None,
        video_dtype: type = np.uint8) -> np.ndarray:
    """
    Create a video array of uints from an HDF5 file, applying
    downsampling and a median filter if desired.

    Parameters
    ----------
    input_path: pathlib.Path
        Path to the HDF5 file containing the movie data

    input_hz:
        Frame rate of the input movie in Hz

    output_hz: float
        Frame rate of the output movie in Hz (set lower than input_hz
        if you want to apply downsampling to the movie)

    spatial_filter: Optional[Callable[[np.ndarray], np.ndarray]]
        The function (if any) used to spatially filter frames after
        downsampling. Accepts an np.ndarray that is the input video;
        returns an np.ndarray that is the spatially filtered video.

    n_processors: int
        Number of parallel processes to be used when processing the movie

    quantiles: Tuple[float, float]
        The quantiles to which to clip the movie before writing it to video

    reticle: bool
        If True, add a grid of red lines to the movie to guide the eye

    tmp_dir: Optional[pathlib.Path]
        Scratch directory to use during processing. When applying the median
        filter, the code writes the filtered movie to disk, rather than try
        to keep two copies of the movie in memory. This gives the user the
        option to specify where the scratch copy of the movie is written.
        If None, the scratch movie will be written to the system's default
        scratch space.

    video_dtype: type
        Type to which the video will be cast (must be either
        np.uint8 or np.uint16)

    Returns
    -------
    video_array: np.ndarray
        array of uints representing the video with the appropriate
        spatiotemporal smoothings applied
    """

    with tempfile.TemporaryDirectory(dir=tmp_dir) as this_tmp_dir:
        tmp_h5 = tempfile.mkstemp(dir=this_tmp_dir, suffix='.h5')[1]
        tmp_h5 = pathlib.Path(tmp_h5)
        logger.info(f'writing h5py to {tmp_h5}')

        create_downsampled_video_h5(
            input_path, input_hz,
            tmp_h5, output_hz,
            spatial_filter,
            n_processors)

        logger.info(f'wrote temp h5py to {tmp_h5}')

        (min_val,
         max_val) = _min_max_from_h5(tmp_h5, quantiles)

        video_array = _video_array_from_h5(
                tmp_h5,
                min_val=min_val,
                max_val=max_val,
                reticle=reticle,
                video_dtype=video_dtype)

        tmp_h5.unlink()

        return video_array


def apply_downsampled_mean_filter_to_video(
        video: np.ndarray,
        kernel_size: int) -> np.ndarray:
    """
    Use skimage.measure.block_reduce to downsample a video
    spatially by taking the mean of (kernel_size, kernel_size)
    blocks

    Parameters
    ----------
    video: np.ndarray
        (ntime, nrows, ncols)

    kernel_size: int
        The side length of the square kernel

    Returns
    -------
    downsampled_video: np.ndarray
        (ntime, ceil(nrows/kernel_size), ceil(n_cols/kernel_size))

    Notes
    -----
    If spatial dimensions are not exactly divisible by kernel_size,
    the input array will be padded by zeros before the mean is taken
    """

    downsampled_video = skimage_measure.block_reduce(
                            video,
                            block_size=(1, kernel_size, kernel_size),
                            func=np.mean,
                            cval=0.0,
                            func_kwargs=None)
    return downsampled_video
