import h5py
import numpy as np
from typing import Tuple, List
from scipy.ndimage.filters import median_filter
from ophys_etl.utils.array_utils import normalize_array


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
