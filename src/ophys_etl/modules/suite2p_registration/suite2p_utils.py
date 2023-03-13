import logging
import warnings
from itertools import product
from time import time

import h5py
import numpy as np
from scipy.stats import sigmaclip
from suite2p.registration.register import (
    pick_initial_reference,
    register_frames,
)
from suite2p.registration.rigid import (
    apply_masks,
    compute_masks,
    phasecorr,
    phasecorr_reference,
    shift_frame,
)


def load_initial_frames(
    file_path: str,
    h5py_key: str,
    n_frames: int,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
) -> np.ndarray:
    """Load a subset of frames from the hdf5 data specified by file_path.

    Only loads frames between trim_frames_start and n_frames - trim_frames_end
    from the movie. If both are 0, load frames from the full movie.

    Parameters
    ----------
    file_path : str
        Location of the raw ophys, HDF5 data to load.
    h5py_key : str
        Name of the dataset to load from the HDF5 file.
    n_frames : int
        Number of frames to load from the input HDF5 data.

    Returns
    -------
    frames : array-like, (n_frames, nrows, ncols)
        Frames selected from the input raw data linearly spaced in index of the
        time axis. If n_frames > tot_frames, a number of frames equal to
        tot_frames is returned.
    """
    with h5py.File(file_path, "r") as hdf5_file:
        # Load all frames as fancy indexing is slower than loading the full
        # data.
        max_frame = hdf5_file[h5py_key].shape[0] - trim_frames_end
        frame_window = hdf5_file[h5py_key][trim_frames_start:max_frame]
        # Total number of frames in the movie.
        tot_frames = frame_window.shape[0]
        requested_frames = np.linspace(
            0, tot_frames, 1 + min(n_frames, tot_frames), dtype=int
        )[:-1]
        frames = frame_window[requested_frames]
    return frames


def compute_reference(
    input_frames: np.ndarray,
    niter: int,
    maxregshift: float,
    smooth_sigma: float,
    smooth_sigma_time: float,
    mask_slope_factor: float = 3,
    logger: callable = None,
) -> np.ndarray:
    """Computes a stacked reference image from the input frames.

    Modified version of Suite2P's compute_reference function with no updating
    of input frames. Picks initial reference then iteratively aligns frames to
    create reference. This code does not reproduce the pre-processing suite2p
    does to data from 1Photon scopes. As such, if processing 1Photon data, the
    user should use the suite2p reference image creation.

    Parameters
    ----------
    input_frames : array-like, (n_frames, nrows, ncols)
        Set of frames to create a reference from.
    niter : int
        Number of iterations to perform when creating the reference image.
    maxregshift : float
        Maximum shift allowed as a fraction of the image width or height, which
        ever is longer.
    smooth_sigma : float
        Width of the Gaussian used to smooth the phase correlation between the
        reference and the frame with which it is being registered.
    smooth_sigma_time : float
        Width of the Gaussian used to smooth between multiple frames by before
        phase correlation.
    mask_slope_factor : int
        Factor to multiply ``smooth_sigma`` by when creating masks for the
        reference image during suite2p phase correlation. These masks down
        weight edges of the image. The default used in suite2p, where this
        method is adapted from, is 3.

    Returns
    -------
    refImg : array-like, (nrows, ncols)
        Reference image created from the input data.
    """
    # Get the dtype of the input frames to properly cast the final reference
    # image as the same type.
    frames_dtype = input_frames.dtype

    # Get initial reference image from suite2p.
    frames = remove_extrema_frames(input_frames)
    ref_image = pick_initial_reference(frames)

    # Determine how much to pad our frames by before shifting to prevent
    # wraps.
    pad_y = int(np.ceil(maxregshift * ref_image.shape[0]))
    pad_x = int(np.ceil(maxregshift * ref_image.shape[1]))

    for idx in range(niter):
        # Compute the number of frames to select in creating the reference
        # image. At most we select half to the input frames.
        nmax = int(frames.shape[0] * (1.0 + idx) / (2 * niter))

        # rigid Suite2P phase registration.
        ymax, xmax, cmax = phasecorr(
            data=apply_masks(
                frames,
                *compute_masks(
                    refImg=ref_image,
                    maskSlope=3 * smooth_sigma,
                ),
            ),
            cfRefImg=phasecorr_reference(
                refImg=ref_image,
                smooth_sigma=smooth_sigma,
            ),
            maxregshift=maxregshift,
            smooth_sigma_time=smooth_sigma_time,
        )

        # Find the indexes of the frames that are the most correlated and
        # select the first nmax.
        isort = np.argsort(-cmax)[:nmax]

        # Copy the most correlated frames so we don't shift the original data.
        # We pad this data to prevent wraps from showing up in the reference
        # image. We pad with NaN values to enable us to use nanmean and only
        # average those pixels that contain data in the average.
        max_corr_frames = np.pad(
            array=frames[isort].astype(float),
            pad_width=((0, 0), (pad_y, pad_y), (pad_x, pad_x)),
            constant_values=np.nan,
        )
        max_corr_xmax = xmax[isort]
        max_corr_ymax = ymax[isort]
        # Apply shift to the copy of the frames.
        for frame, dy, dx in zip(
            max_corr_frames, max_corr_ymax, max_corr_xmax
        ):
            frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)

        # Create a new reference image from the highest correlated data.
        with warnings.catch_warnings():
            # Assuming the motion correction went well, there should be a lot
            # of empty values in the padded area around the frames. We suppress
            # warnings for these "Empty Slices" as they are expected.
            warnings.filterwarnings("ignore", "Mean of empty slice")
            ref_image = np.nanmean(max_corr_frames, axis=0)
        # Shift reference image to position of mean shifts to remove any bulk
        # displacement.
        ref_image = shift_frame(
            frame=ref_image,
            dy=int(np.round(-max_corr_ymax.mean())),
            dx=int(np.round(-max_corr_xmax.mean())),
        )
        # Clip the reference image back down to the original size and remove
        # any NaNs remaining. Throw warning if a NaN is found.
        ref_image = ref_image[pad_y:-pad_y, pad_x:-pad_x]
        if np.any(np.isnan(ref_image)):
            # NaNs can sometimes be left over from the image padding during the
            # first few iterations before the reference image has converged.
            # If there are still NaNs left after the final iteration, we
            # throw the following warning.
            if idx + 1 == niter:
                logging.warning(
                    f"Warning: {np.isnan(ref_image).sum()} NaN pixels were "
                    "found in the reference image on the final iteration. "
                    "Likely the image quality is low and shifting frames "
                    "failed. Setting NaN values to the image mean."
                )
            ref_image = np.nan_to_num(
                ref_image, nan=np.nanmean(ref_image), copy=False
            )
        ref_image = ref_image.astype(frames_dtype)

    return ref_image


def remove_extrema_frames(
    input_frames: np.ndarray, n_sigma: float = 3
) -> np.ndarray:
    """Remove frames with extremum mean values from the frames used in
    reference image processing/creation.

    Likely these are empty frames of pure noise or very high intensity frames
    relative to mean.

    Parameters
    ----------
    input_frames : numpy.ndarray, (N, M, K)
        Set of frames to trim.
    n_sigma : float, optional
        Number of standard deviations to above which to clip. Default is 3
        which was found to remove all empty frames while preserving most
        frames.

    Returns
    -------
    trimmed_frames : numpy.ndarray, (N, M, K)
        Set of frames with the extremum frames removed.
    """
    frame_means = np.mean(input_frames, axis=(1, 2))
    _, low_cut, high_cut = sigmaclip(frame_means, low=n_sigma, high=n_sigma)
    trimmed_frames = input_frames[
        np.logical_and(frame_means > low_cut, frame_means < high_cut)
    ]
    return trimmed_frames


def optimize_motion_parameters(
    initial_frames: np.ndarray,
    smooth_sigmas: np.array,
    smooth_sigma_times: np.array,
    suite2p_args: dict,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
    n_batches: int = 20,
    logger: callable = None,
) -> dict:
    """Loop over a range of parameters and select the best set from the
    max acutance of the final, average image.

    Parameters
    ----------
    initial_frames : numpy.ndarray, (N, M, K)
        Smaller subset of frames to create a reference image from.
    smooth_sigmas : numpy.ndarray, (N,)
        Array of suite2p smooth sigma values to attempt. Number of iterations
        will be len(`smooth_sigmas`) * len(`smooth_sigma_times`).
    smooth_sigma_times : numpy.ndarray, (N,)
        Array of suite2p smooth sigma time values to attempt. Number of
        iterations will be len(`smooth_sigmas`) * len(`smooth_sigma_times`).
    suite2p_args : dict
        A dictionary of suite2p configs containing at minimum:

        ``"h5py"``
            HDF5 file containing to the movie to motion correct.
        ``"h5py_key"``
            Name of the dataset where the movie to be motion corrected is
            stored.
        ``"maxregshift"``
            Maximum shift allowed as a fraction of the image dimensions.
    trim_frames_start : int, optional
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_start : int, optional
        Number of frames to disregard from the end of the movie. Default 0.
    n_batches : int
        Number of batches to load. Processing a large number of frames at once
        will likely result in running out of memory, hence processing in
        batches. Total returned size isn_batches * suit2p_args['batch_size'].
    logger : callable, optional
        Function to print to stdout or a log.

    Returns
    -------
    best_result : dict
        A dict containing the final results of the search:

        ``ave_image``
            Image created with the settings yielding the highest image acutance
            (numpy.ndarray, (N, M))
        ``ref_image``
            Reference Image created with the settings yielding the highest
            image acutance (numpy.ndarray, (N, M))
        ``acutance``
            Acutance of ``best_image``. (float)
        ``smooth_sigma``
            Value of ``smooth_sigma`` found to yield the best acutance (float).
        ``smooth_sigma_time``
            Value of ``smooth_sigma_time`` found to yield the best acutance
            (float).
    """
    best_results = {
        "acutance": 1e-16,
        "ave_image": np.array([]),
        "ref_image": np.array([]),
        "smooth_sigma": -1,
        "smooth_sigma_time": -1,
    }
    logger("Starting search for best smoothing parameters...")
    sub_frames = load_representative_sub_frames(
        suite2p_args["h5py"],
        suite2p_args["h5py_key"],
        trim_frames_start,
        trim_frames_end,
        n_batches=n_batches,
        batch_size=suite2p_args["batch_size"],
    )
    start_time = time()
    for param_spatial, param_time in product(
        smooth_sigmas, smooth_sigma_times
    ):
        current_args = suite2p_args.copy()
        current_args["smooth_sigma"] = param_spatial
        current_args["smooth_sigma_time"] = param_time

        if logger:
            logger(
                f'\tTrying: smooth_sigma={current_args["smooth_sigma"]}, '
                f'smooth_sigma_time={current_args["smooth_sigma_time"]}'
            )

        ref_image = compute_reference(
            initial_frames,
            8,
            current_args["maxregshift"],
            current_args["smooth_sigma"],
            current_args["smooth_sigma_time"],
        )
        image_results = create_ave_image(
            ref_image,
            sub_frames.copy(),
            current_args,
            batch_size=suite2p_args["batch_size"],
        )
        ave_image = image_results["ave_image"]
        # Compute the acutance ignoring the motion border. Sharp motion
        # borders can potentially get rewarded with high acutance.
        current_acu = compute_acutance(
            ave_image,
            image_results["min_y"],
            image_results["max_y"],
            image_results["min_x"],
            image_results["max_x"],
        )

        if current_acu > best_results["acutance"]:
            best_results["acutance"] = current_acu
            best_results["ave_image"] = ave_image
            best_results["ref_image"] = ref_image
            best_results["smooth_sigma"] = current_args["smooth_sigma"]
            best_results["smooth_sigma_time"] = current_args[
                "smooth_sigma_time"
            ]
        if logger:
            logger(f"\t\tResulting acutance={current_acu:.4f}")
    if logger:
        logger(
            f"Found best motion parameters in {time() - start_time:.0f} "
            f'seconds, with image acutance={best_results["acutance"]:.4f}, '
            f'for parameters: smooth_sigma={best_results["smooth_sigma"]}, '
            f'smooth_sigma_time={best_results["smooth_sigma_time"]}'
        )
    return best_results


def load_representative_sub_frames(
    h5py_name,
    h5py_key,
    trim_frames_start: int = 0,
    trim_frames_end: int = 0,
    n_batches: int = 20,
    batch_size: int = 500,
):
    """Load a subset of frames spanning the full movie.

    Parameters
    ----------
    h5py_name : str
        Path to the h5 file to load frames from.
    h5py_key : str
        Name of the h5 dataset containing the movie.
    trim_frames_start : int, optional
        Number of frames to disregard from the start of the movie. Default 0.
    trim_frames_start : int, optional
        Number of frames to disregard from the end of the movie. Default 0.
    n_batches : int
        Number of batches to load. Total returned size is
        n_batches * batch_size.
    batch_size : int, optional
        Number of frames to process at once. Total returned size is
        n_batches * batch_size.

    Returns
    -------
    """
    output_frames = []
    frame_fracts = np.arange(0, 1, 1 / n_batches)
    with h5py.File(h5py_name, "r") as h5_file:
        dataset = h5_file[h5py_key]
        total_frames = dataset.shape[0] - trim_frames_start - trim_frames_end
        if total_frames < n_batches * batch_size:
            return dataset[:]
        for percent_start in frame_fracts:
            frame_start = int(percent_start * total_frames + trim_frames_start)
            output_frames.append(
                dataset[frame_start:frame_start + batch_size]
            )
    return np.concatenate(output_frames)


def create_ave_image(
    ref_image: np.ndarray,
    input_frames: np.ndarray,
    suite2p_args: dict,
    batch_size: int = 500,
) -> dict:
    """Run suite2p image motion correction over a full movie.

    Parameters
    ----------
    ref_image : numpy.ndarray, (N, M)
        Reference image to correlate with movie frames.
    input_frames : numpy.ndarray, (L, N, M)
        Frames to motion correct and compute average image/acutance of.
    suite2p_args : dict
        Dictionary of suite2p args containing:

        ``"h5py"``
            HDF5 file containing to the movie to motion correct.
        ``"h5py_key"``
            Name of the dataset where the movie to be motion corrected is
            stored.
        ``"maxregshift"``
            Maximum shift allowed as a fraction of the image dimensions.
        ``"smooth_sigma"``
            Spatial Gaussian smoothing parameter used by suite2p to smooth
            frames before correlation. (float).
        ``"smooth_sigma_time"``
            Time Gaussian smoothing of frames to apply before correlation.
            (float).
    batch_size : int, optional
        Number of frames to process at once.

    Returns
    -------
    ave_image_dict : dict
        A dict containing the average image and motion border values:

        ``ave_image``
            Image created with the settings yielding the highest image acutance
            (numpy.ndarray, (N, M))
        ``min_y``
            Minimum y allowed value in image array. Below this is motion
            border.
        ``max_y``
            Maximum y allowed value in image array. Above this is motion
            border.
        ``min_x``
            Minimum x allowed value in image array. Below this is motion
            border.
        ``max_x``
            Maximum x allowed value in image array. Above this is motion
            border.
    """
    ave_frame = np.zeros((ref_image.shape[0], ref_image.shape[1]))
    min_y = 0
    max_y = 0
    min_x = 0
    max_x = 0
    tot_frames = input_frames.shape[0]
    add_modify_required_parameters(suite2p_args)
    for start_idx in np.arange(0, tot_frames, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > tot_frames:
            end_idx = tot_frames
        frames = input_frames[start_idx:end_idx]
        frames, dy, dx, _, _, _, _ = register_frames(
            refAndMasks=ref_image, frames=frames, ops=suite2p_args
        )
        min_y = min(min_y, dy.min())
        max_y = max(max_y, dy.max())
        min_x = min(min_x, dx.min())
        max_x = max(max_x, dx.max())
        ave_frame += frames.sum(axis=0) / tot_frames

    return {
        "ave_image": ave_frame,
        "min_y": int(np.fabs(min_y)),
        "max_y": int(max_y),
        "min_x": int(np.fabs(min_x)),
        "max_x": int(max_x),
    }


def add_modify_required_parameters(suite2p_args: dict):
    """Check that minimum parameters needed by suite2p registration are
    available. If not add them to the suite2p_args dict.

    Additionally, make sure that nonrigid is set to false as are gridsearch
    of parameters above is not setup to use nonrigid.

    Parameters
    ----------
    suite2p_args : dict
        Suite2p ops dictionary with potentially missing values.
    """
    if suite2p_args.get("1Preg") is None:
        suite2p_args["1Preg"] = False
    if suite2p_args.get("bidiphase") is None:
        suite2p_args["bidiphase"] = False
    if suite2p_args.get("nonrigid") is None:
        suite2p_args["nonrigid"] = False
    if suite2p_args.get("norm_frames") is None:
        suite2p_args["norm_frames"] = True
    # Don't use nonrigid for parameter search.
    suite2p_args["nonrigid"] = False


def compute_acutance(
    image: np.ndarray,
    min_cut_y: int = 0,
    max_cut_y: int = 0,
    min_cut_x: int = 0,
    max_cut_x: int = 0,
) -> float:
    """Compute the acutance (sharpness) of an image.

    Parameters
    ----------
    image : numpy.ndarray, (N, M)
        Image to compute acutance of.
    min_cut_y : int
        Number of pixels to cut from the beginning of the y axis.
    max_cut_y : int
        Number of pixels to cut from the end of the y axis.
    min_cut_x : int
        Number of pixels to cut from the beginning of the x axis.
    max_cut_x : int
        Number of pixels to cut from the end of the x axis.

    Returns
    -------
    acutance : float
        Acutance of the image.
    """
    im_max_y, im_max_x = image.shape

    cut_image = image[
        min_cut_y:im_max_y - max_cut_y, min_cut_x:im_max_x - max_cut_x
    ]
    grady, gradx = np.gradient(cut_image)
    return (grady**2 + gradx**2).mean()
