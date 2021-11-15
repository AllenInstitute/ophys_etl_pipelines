import numpy as np
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
