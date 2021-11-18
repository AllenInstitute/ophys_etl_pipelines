import pathlib
import h5py
import numpy as np
import multiprocessing
import multiprocessing.managers
import scipy.ndimage as scipy_ndimage


def n_frames_from_hz(
        input_frame_rate: float,
        downsampled_frame_rate: float) -> int:
    """
    Find the number of frames to group together to downsample
    a video from input_frame_rate to downsampled_frame_rate

    Parameters
    ----------
    input_frame_rate: float

    downsampled_frame_rate: float

    Returns
    -------
    frames_to_group: int

    Notes
    -----
    If input_frame_rate/downsampled_frame_rate < 1, will return 1
    """

    frames_to_group = np.round(input_frame_rate/downsampled_frame_rate)
    frames_to_group = frames_to_group.astype(int)
    return max(1, frames_to_group)


def apply_median_filter_to_video(
        video: np.ndarray,
        kernel_size: int) -> np.ndarray:
    """
    Apply a median filter to a subset of frames

    Parameters
    ----------
    video: np.ndarray
        array of shape (ntime, nrows, ncols)

    kernel_size: int
        side length of square kernel to use for median filter

    Returns
    -------
    filtered_frames: np.ndarray
        (ntime, nrows, ncols) with each frame passed through
        the median filter
    """

    filtered_frames = np.zeros(video.shape,
                               dtype=float)

    for i_frame in range(filtered_frames.shape[0]):
        filtered_frames[i_frame, :, :] = scipy_ndimage.median_filter(
                                             video[i_frame, :, :],
                                             size=kernel_size,
                                             mode='reflect')
    return filtered_frames


def decimate_video(
        video: np.ndarray,
        frames_to_group: int) -> np.ndarray:
    """
    Take a video, sum every frames_to_group together
    and take the mean. Return a video representing
    the mean of every frames_to_group chunk (i.e. if the input
    chunk of frames contains 100 frames and frames_to_group=20,
    the output video will contain 5 frames

    Parameters
    ----------
    video: np.ndarray
        (ntime, nrows, ncols)

    frames_to_avg: int
        Number of frames to group together and mean when decimating

    Returns
    -------
    decimated_video: np.ndarray
    """

    ntime_in = video.shape[0]
    ntime_out = np.ceil(ntime_in/frames_to_group).astype(int)

    decimated_video = np.zeros((ntime_out,
                                video.shape[1],
                                video.shape[2]),
                               dtype=float)

    for i_out, i0 in enumerate(range(0, ntime_in, frames_to_group)):
        i1 = min(ntime_in, i0+frames_to_group)
        decimated_video[i_out, :, :] = np.mean(video[i0:i1, :, :], axis=0)

    return decimated_video


def filter_worker(video: np.ndarray,
                  kernel_size: int,
                  output_list: multiprocessing.managers.ListProxy) -> None:
    """
    Worker method to apply apply_median_filter_to_video to a subset of
    video frames from a video

    Parameters
    ----------
    video: np.ndarray
        (ntime, nrows, ncolumns)

    kernel_size: int

    output_list: multiprocessing.managers.ListProxy
        List (shared across processes) where median filtered
        frames will be kept. Because it does not matter the order of
        frames when they are cast into a maximum projection image,
        the order of the frames does not need to be preserved here.
    """
    local_result = apply_median_filter_to_video(video, kernel_size)
    output_list.append(local_result)


def median_filtered_max_projection_from_array(
        video: np.ndarray,
        input_frame_rate: float,
        downsampled_frame_rate: float,
        median_filter_kernel_size: int,
        n_processors: int) -> np.ndarray:
    """
    Generate a maximum projection from a video by
    1) downsampling the movie from input_frame_rate to downsampled_frame_rate
    2) applying a median filter to every frame of the downsampled video
    3) taking the maximum of the downsampled, filtered video at each pixel

    Parameters
    ----------
    video: np.ndarray
        (ntime, nrows, ncols)

    input_frame_rate: float

    downsampled_frame_rate: float
        In the same units as input_frame_rate (ratio is all that matters)

    median_filter_kernel_size: int
        The side length of the square kernel used when applying the median
        filter.

    n_processors:
        The number of parallel processes available (used when applying
        the median filter)

    Returns
    -------
    maximum_projection: np.ndarray
        (nrows, ncols)
    """

    frames_to_group = n_frames_from_hz(
                            input_frame_rate,
                            downsampled_frame_rate)

    if frames_to_group > 1:
        video = decimate_video(video, frames_to_group)

    n_frames_per_chunk = np.ceil(video.shape[0]/n_processors).astype(int)
    n_frames_per_chunk = max(1, n_frames_per_chunk)
    process_list = []
    mgr = multiprocessing.Manager()
    output_list = mgr.list()
    ntime0 = video.shape[0]
    for i0 in range(0, ntime0, n_frames_per_chunk):

        # in order to avoid holding too many copies of the
        # (large) video in memory at once, "destroy" video
        # as it is processed
        input_chunk = video[:n_frames_per_chunk, :, :]
        video = video[n_frames_per_chunk:, :, :]

        p = multiprocessing.Process(
                    target=filter_worker,
                    args=(input_chunk,
                          median_filter_kernel_size,
                          output_list))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    return np.vstack(output_list).max(axis=0)


def median_filtered_max_projection_from_path(
        video_path: pathlib.Path,
        input_frame_rate: float,
        downsampled_frame_rate: float,
        median_filter_kernel_size: int,
        n_processors: int,
        n_frames_at_once: int = 10000) -> np.ndarray:
    """
    Generate a maximum projection from an image by
    1) downsampling the movie from input_frame_rate to downsampled_frame_rate
    2) applying a median filter to every frame of the downsampled video
    3) taking the maximum of the downsampled, filtered video at each pixel

    Parameters
    ----------
    video_path: pathlib.Path
        Path to HDF5 file containing vidoe data (data must be keyed to 'data')

    input_frame_rate: float

    downsampled_frame_rate: float
        In the same units as input_frame_rate (ratio is all that matters)

    median_filter_kernel_size: int
        The side length of the square kernel used when applying the median
        filter.

    n_processors:
        The number of parallel processes available (used when applying
        the median filter)

    n_frames_at_once: int
        The decimation and median filter steps can indirectly cause
        multiple copies of the video to be held in memory at once.
        This parameter limits the number of frames that are loaded
        from the HDF5 file at one time (if <1, load all frames at once)

    Returns
    -------
    max_projection_image: np.ndarray
    """

    with h5py.File(video_path, 'r') as in_file:
        n_total_frames = in_file['data'].shape[0]

    if n_frames_at_once <= 0:
        n_frames_at_once = n_total_frames
    else:
        frames_to_group = n_frames_from_hz(
                                input_frame_rate,
                                downsampled_frame_rate)

        n = np.round(n_frames_at_once/frames_to_group).astype(int)
        n_frames_at_once = n*frames_to_group

    sub_img_list = []
    for frame0 in range(0, n_total_frames, n_frames_at_once):
        frame1 = min(n_total_frames, frame0+n_frames_at_once)
        with h5py.File(video_path, 'r') as in_file:
            video_data = in_file['data'][frame0:frame1, :, :]

        img = median_filtered_max_projection_from_array(
                video_data,
                input_frame_rate,
                downsampled_frame_rate,
                median_filter_kernel_size,
                n_processors)

        sub_img_list.append(img)

    img = np.stack(sub_img_list).max(axis=0)
    return img
