import pathlib
import h5py
import numpy as np
import multiprocessing
import multiprocessing.managers
import scipy.ndimage as scipy_ndimage


def filter_chunk_of_frames(
        chunk_of_frames: np.ndarray,
        kernel_size: int) -> np.ndarray:
    """
    Apply a median filter to a subset of frames

    Parameters
    ----------
    chunk_of_frames: np.ndarray
        array of shape (ntime, nrows, ncols)

    kernel_size: int
        side length of square kernel to use for median filter

    Returns
    -------
    filtered_frames: np.ndarray
        (ntime, nrows, ncols) with each frame passed through
        the median filter
    """

    filtered_frames = np.zeros(chunk_of_frames.shape,
                               dtype=float)

    for i_frame in range(filtered_frames.shape[0]):
        filtered_frames[i_frame, :, :] = scipy_ndimage.median_filter(
                                             chunk_of_frames[i_frame, :, :],
                                             size=kernel_size,
                                             mode='reflect')
    return filtered_frames


def decimate_video(
        video: np.ndarray,
        frames_to_group: int) -> np.ndarray:
    """
    Take a video, sum every frames_to_group together
    and take the mean. Return a video representing
    the mean of every frames_to_avg chunk (i.e. if the input
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


def decimated_video_from_path(
        video_path: pathlib.Path,
        frame0: int,
        frame1: int,
        frames_to_group: int,
        video_lock):

    with video_lock:
        with h5py.File(video_path, 'r') as in_file:
            video = in_file['data'][frame0:frame1, :, :]

    if frames_to_group > 1:
        return decimate_video(video, frames_to_group)
    return video


def filter_worker(video_path: pathlib.Path,
                  frame0: int,
                  frame1: int,
                  frames_to_group: int,
                  kernel_size: int,
                  output_list: multiprocessing.managers.ListProxy,
                  video_lock) -> None:
    """
    Worker method to apply filter_chunk_of_frames to a subset of
    video frames from decimated video

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
    video = decimated_video_from_path(
                video_path,
                frame0,
                frame1,
                frames_to_group,
                video_lock)

    local_result = filter_chunk_of_frames(video, kernel_size)
    local_result = np.max(local_result, axis=0)
    output_list.append(local_result)


def generate_max_projection(
        video_path: pathlib.Path,
        input_frame_rate: float,
        downsampled_frame_rate: float,
        median_filter_kernel_size: int,
        n_processors: int) -> np.ndarray:
    """
    Generate a maximum projection from an image by
    1) downsampling the movie from input_frame_rate to downsampled_frame_rate
    2) applying a median filter to every frame of the downsampled video
    3) taking the maximum of the downsampled, filtered video at each pixel

    Parameters
    ----------
    video_path: pathlib.Path

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
    with h5py.File(video_path, 'r') as in_file:
        n_frames_total = in_file['data'].shape[0]

    frames_to_group = np.round(input_frame_rate/downsampled_frame_rate)
    frames_to_group = frames_to_group.astype(int)

    n_frames_per_chunk = np.ceil(n_frames_total/n_processors).astype(int)
    # set n_frames_per_chunk to a multiple of frames_to_group
    n = np.ceil(n_frames_per_chunk/frames_to_group).astype(int)
    n_frames_per_chunk = frames_to_group*n

    process_list = []
    mgr = multiprocessing.Manager()
    output_list = mgr.list()
    video_lock = mgr.Lock()
    for frame0 in range(0, n_frames_total, n_frames_per_chunk):
        frame1 = min(n_frames_total, frame0+n_frames_per_chunk)

        p = multiprocessing.Process(
                    target=filter_worker,
                    args=(video_path,
                          frame0,
                          frame1,
                          frames_to_group,
                          median_filter_kernel_size,
                          output_list,
                          video_lock))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    return np.stack(output_list).max(axis=0)


def scale_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Scale an image of arbitrary type to uint8

    Parameters
    ----------
    img: np.ndarray

    Returns
    -------
    scaled_img: np.ndarray

    Notes:
    ------
    Image will have the minimum subtracted and then the full
    dynamic range will be scaled into the range [0, 255]
    """

    mn = img.min()
    img = img-mn
    return np.round(255.0*img.astype(float)/img.max()).astype(np.uint8)
