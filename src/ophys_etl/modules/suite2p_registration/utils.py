import h5py
import numpy as np
from typing import Callable, Tuple, List
from scipy.ndimage.filters import median_filter
from ophys_etl.utils.array_utils import normalize_array
from suite2p.registration.rigid import shift_frame


def check_and_warn_on_datatype(h5py_name: str,
                               h5py_key: str,
                               logger: Callable):
    """Suite2p assumes int16 types throughout code. Check that the input
    data is type int16 else throw a warning.

    Parameters
    ----------
    h5py_name : str
        Path to the HDF5 containing the data.
    h5py_key : str
        Name of the dataset to check.
    logger : Callable
        Logger to output logger warning to.
    """
    with h5py.File(h5py_name, 'r') as h5_file:
        dataset = h5_file[h5py_key]

        if dataset.dtype.byteorder == '>':
            logger('Data byteorder is big-endian which may cause issues in '
                   'suite2p. This may result in a crash or unexpected '
                   'results.')
        if dataset.dtype.name != 'int16':
            logger(f'Data type is {dataset.dtype.name} and not int16. Suite2p '
                   'assumes int16 data as input and throughout codebase. '
                   'Non-int16 data may result in unexpected results or '
                   'crashes.')


def find_movie_start_end_empty_frames(
        h5py_name: str,
        h5py_key: str,
        n_sigma: float = 5,
        logger: callable = None) -> Tuple[int, int]:
    """Load a movie from HDF5 and find frames at the start and end of the
    movie that are empty or pure noise and 5 sigma discrepant from the
    average frame.

    If a non-contiguous set of frames is found, the code will return 0 for
    that half of the movie and throw a warning about the quality of the data.

    Parameters
    ----------
    h5py_name : str
        Name of the HDF5 file to load from.
    h5py_key : str
        Name of the dataset to load from the HDF5 file.
    n_sigma : float
        Number of standard deviations beyond which a frame is considered an
        outlier and "empty".
    logger : callable, optional
        Function to print warning messages to.

    Returns
    -------
    trim_frames : Tuple[int, int]
        Tuple of the number of frames to cut from the start and end of the
        movie as (n_trim_start, n_trim_end).
    """
    # Load the data.
    with h5py.File(h5py_name, 'r') as h5_file:
        frames = h5_file[h5py_key][:]
    # Find the midpoint of the movie.
    midpoint = frames.shape[0] // 2

    # We discover empty or extrema frames by comparing the mean of each frames
    # to the mean of the full movie.
    means = frames.mean(axis=(1, 2))
    mean_of_frames = means.mean()

    # Compute a robust standard deviation that is not sensitive to the
    # outliers we are attempting to find.
    quart_low, quart_high = np.percentile(means, [25, 75])
    # Convert the inner quartile range to an estimate of the standard deviation
    # 0.6745 is the converting factor between the inner quartile and a
    # traditional standard deviation.
    std_est = (quart_high - quart_low) / (2 * 0.6745)

    # Get the indexes of the frames that are found to be n_sigma deviating.
    start_idxs = np.sort(
        np.argwhere(
            means[:midpoint] < mean_of_frames - n_sigma * std_est)).flatten()
    end_idxs = np.sort(
        np.argwhere(
            means[midpoint:] < mean_of_frames - n_sigma * std_est)).flatten() \
        + midpoint

    # Get the total number of these frames.
    lowside = len(start_idxs)
    highside = len(end_idxs)

    # Check to make sure that the indexes found were only from the start/end
    # of the movie. If not, throw a warning and reset the number of frames
    # found to zero.
    if not np.array_equal(start_idxs,
                          np.arange(0, lowside, dtype=start_idxs.dtype)):
        lowside = 0
        if logger is not None:
            logger(f"{n_sigma} sigma discrepant frames found outside the "
                   "beginning of the movie. Please inspect the movie for data "
                   "quality. Not trimming frames from the movie beginning.")
    if not np.array_equal(end_idxs,
                          np.arange(frames.shape[0] - highside,
                                    frames.shape[0],
                                    dtype=end_idxs.dtype)):
        highside = 0
        if logger is not None:
            logger(f"{n_sigma} sigma discrepant frames found outside the end "
                   "of the movie. Please inspect the movie for data quality. "
                   "Not trimming frames from the movie end.")

    return (lowside, highside)


def reset_frame_shift(frames: np.ndarray,
                      dy_array: np.ndarray,
                      dx_array: np.ndarray,
                      trim_frames_start: int,
                      trim_frames_end: int):
    """Reset the frames of a movie and their shifts.

    Shifts the frame back to its original location and resets the shifts for
    those frames to (0, 0). Frames, dy_array, and dx_array are edited in
    place.

    Parameters
    ----------
    frames : numpy.ndarray, (N, M, K)
        Full movie to reset frames in.
    dy_array : numpy.ndarray, (N,)
        Array of shifts in the y direction for each frame of the movie.
    dx_array : numpy.ndarray, (N,)
        Array of shifts in the x direction for each frame of the movie.
    trim_frames_start : int
        Number of frames at the start of the movie that were identified as
        empty or pure noise.
    trim_frames_end : int
        Number of frames at the end of the movie that were identified as
        empty or pure noise.
    """
    for idx in range(trim_frames_start):
        dy = -dy_array[idx]
        dx = -dx_array[idx]
        frames[idx] = shift_frame(frames[idx], dy, dx)
        dy_array[idx] = 0
        dx_array[idx] = 0

    for idx in range(frames.shape[0] - trim_frames_end, frames.shape[0]):
        dy = -dy_array[idx]
        dx = -dx_array[idx]
        frames[idx] = shift_frame(frames[idx], dy, dx)
        dy_array[idx] = 0
        dx_array[idx] = 0


def projection_process(data: np.ndarray,
                       projection: str = "max") -> np.ndarray:
    """

    Parameters
    ----------
    data: np.ndarray
        nframes x nrows x ncols, uint16
    projection: str
        "max" or "avg"

    Returns
    -------
    proj: np.ndarray
        nrows x ncols, uint8

    """
    if projection == "max":
        proj = np.max(data, axis=0)
    elif projection == "avg":
        proj = np.mean(data, axis=0)
    else:
        raise ValueError("projection can be \"max\" or \"avg\" not "
                         f"{projection}")
    return normalize_array(proj)


def identify_and_clip_outliers(data: np.ndarray,
                               med_filter_size: int,
                               thresh: int) -> Tuple[np.ndarray, List]:
    """given data, identify the indices of outliers
    based on median filter detrending, and a threshold

    Parameters
    ----------
    data: np.ndarray
        1D array of samples
    med_filter_size: int
        the number of samples for 'size' in
        scipy.ndimage.filters.median_filter
    thresh: int
        multipled by the noise estimate to establish a threshold, above
        which, samples will be marked as outliers.

    Returns
    -------
    data: np.ndarry
        1D array of samples, clipped to threshold around median-filtered data
    indices: np.ndarray
        the indices where clipping took place

    """
    data_filtered = median_filter(data, med_filter_size, mode='nearest')
    detrended = data - data_filtered
    indices = np.argwhere(np.abs(detrended) > thresh).flatten()
    data[indices] = np.clip(data[indices],
                            data_filtered[indices] - thresh,
                            data_filtered[indices] + thresh)
    return data, indices


def check_movie_against_raw(corr_data, raw_hdf5, h5py_key, batch_size=2000):
    """Check that the values stored in all frames match between motion
    corrected and raw movies.

    Test is not run if the suite2p option nonrigid is used as this option
    does not preserve flux.

    Parameters
    ----------
    corr_data : numpy.ndarray, (N, M, L)
        Motion corrected movie.
    raw_hdf5 : str
        Full path location to raw movie.
    h5py_key : str
        Name of the dataset containing the raw data.
    batch_size : int
        Number of frames to load from the h5py at once.
    """
    with h5py.File(raw_hdf5) as raw_file:
        raw_dataset = raw_file[h5py_key]
        tot_corr_frames = corr_data.shape[0]
        tot_raw_frames = raw_dataset.shape[0]
        if tot_corr_frames != tot_raw_frames:
            raise ValueError('Length of motion corrected movie does '
                             'not match length of raw movie. Something went '
                             'wrong during motion correction. Exiting.')
        # Loop through each chunk of data.
        for start_idx in np.arange(
                0, tot_corr_frames, batch_size, dtype=int):
            end_idx = start_idx + batch_size
            if end_idx > tot_corr_frames:
                end_idx = tot_corr_frames
            raw_frames = raw_dataset[start_idx:end_idx]
            corr_frames = corr_data[start_idx:end_idx]
            # Test each frames values.
            for frame_idx, (raw_frame, corr_frame) in enumerate(zip(
                    raw_frames, corr_frames)):
                # Flatten and sort to test all array values are the same.
                flat_raw = raw_frame.flatten()
                flat_corr = corr_frame.flatten()
                flat_raw.sort()
                flat_corr.sort()
                if not np.array_equal(flat_raw, flat_corr):
                    raise ValueError(
                        'The distribution of pixel values in the raw movie '
                        'does not match the distribution of pixel values in'
                        f'the corrected movie at frame {frame_idx}. '
                        'Something went wrong during motion correction. '
                        'Exiting.')
