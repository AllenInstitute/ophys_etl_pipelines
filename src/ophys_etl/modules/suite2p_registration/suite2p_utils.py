import h5py
import numpy as np
from time import time
from suite2p.registration.register import (pick_initial_reference,
                                           register_frames)
from suite2p.registration.rigid import (apply_masks, compute_masks, phasecorr,
                                        phasecorr_reference, shift_frame)


def load_initial_frames(file_path: str,
                        h5py_key: str,
                        n_frames: int) -> np.ndarray:
    """Load a subset of frames from the hdf5 data specified by file_path.

    Parameters
    ----------
    file_path : str
        Location of the raw 2Photon, HDF5 data to load.
    h5py_key : str
        Name of the dataset to load from the HDF5 file.
    n_frames : int
        Number of frames to load from the input HDF5 data.

    Returns
    -------
    frames : array-like, (n_frames, nrows, ncols)
        Selected frames from the input raw data linearly spaced in index of the
        time axis. If n_frames > tot_frames, a number of frames equal to
        tot_frames is returned.
    """
    with h5py.File(file_path, 'r') as hdf5_file:
        # Load all frames as fancy indexing is slower than loading the full
        # data.
        all_frames = hdf5_file[h5py_key][:]
        # Total number of frames in the movie.
        tot_frames = all_frames.shape[0]
        requested_frames = np.linspace(0,
                                       tot_frames,
                                       1 + min(n_frames, tot_frames),
                                       dtype=int)[:-1]
        frames = all_frames[requested_frames]
    return frames


def compute_reference(frames: np.ndarray,
                      niter: int,
                      maxregshift: float,
                      smooth_sigma: float,
                      smooth_sigma_time: float,
                      mask_slope_factor: float = 3) -> np.ndarray:
    """Computes a stacked reference image from the input frames.

    Modified version of Suite2P's compute_reference function with no updating
    of input frames. Picks initial reference then iteratively aligns frames to
    create reference. This code does not reproduce the pre-processing suite2p
    does to data from 1Photon scopes. As such, if processing 1Photon data, the
    user should use the suite2p reference image creation.

    Parameters
    ----------
    frames : array-like, (n_frames, nrows, ncols)
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
    # Get initial reference image from suite2p.
    ref_image = pick_initial_reference(frames)

    for idx in range(niter):
        # Compute the number of frames to select in creating the reference
        # image. At most we select half to the input frames.
        nmax = int(frames.shape[0] * (1. + idx) / (2 * niter))

        # rigid Suite2P phase registration.
        ymax, xmax, cmax = phasecorr(
            data=apply_masks(frames,
                             *compute_masks(refImg=ref_image,
                                            maskSlope=3 * smooth_sigma,)),
            cfRefImg=phasecorr_reference(
                refImg=ref_image,
                smooth_sigma=smooth_sigma,),
            maxregshift=maxregshift,
            smooth_sigma_time=smooth_sigma_time,
        )

        # Find the indexes of the frames that are the most correlated and
        # select the first nmax.
        isort = np.argsort(-cmax)[:nmax]

        # Copy the most correlated frames so we don't shift the original data.
        max_corr_frames = np.copy(frames[isort])
        max_corr_xmax = xmax[isort]
        max_corr_ymax = ymax[isort]
        # Apply shift to the copy of the frames.
        for frame, dy, dx in zip(max_corr_frames,
                                 max_corr_ymax,
                                 max_corr_xmax):
            frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)

        # Create a new reference image from the highest correlated data.
        ref_image = max_corr_frames.mean(axis=0).astype(np.int16)
        # Shift reference image to position of mean shifts to remove any bulk
        # displacement.
        ref_image = shift_frame(
            frame=ref_image,
            dy=int(np.round(-max_corr_ymax.mean())),
            dx=int(np.round(-max_corr_xmax.mean()))
        )

    return ref_image


def optimize_motion_parameters(initial_frames: np.ndarray,
                               smooth_sigmas: np.array,
                               smooth_sigma_times: np.array,
                               suite2p_args: dict,
                               logger: callable = None) -> dict:
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
    params_spatial, params_time = np.meshgrid(smooth_sigmas,
                                              smooth_sigma_times)

    best_results = {'acutance': 1e-16,
                    'ave_image': np.array([]),
                    'ref_image': np.array([]),
                    'smooth_sigma': -1,
                    'smooth_sigma_time': -1}
    logger('Starting search for best smoothing parameters...')
    start_time = time()
    for param_spatial, param_time in zip(params_spatial.flatten(),
                                         params_time.flatten()):
        current_args = suite2p_args.copy()
        current_args['smooth_sigma'] = param_spatial
        current_args['smooth_sigma_time'] = param_time

        if logger:
            logger(
                f'\tTrying: smooth_sigma={current_args["smooth_sigma"]}, '
                f'smooth_sigma_time={current_args["smooth_sigma_time"]}')

        ref_image = compute_reference(initial_frames,
                                      8,
                                      current_args['maxregshift'],
                                      current_args['smooth_sigma'],
                                      current_args['smooth_sigma_time'])
        image_results = create_ave_image(ref_image, current_args)
        ave_image = image_results['ave_image']
        dy_max = image_results['dy_max']
        dx_max = image_results['dx_max']
        # Compute the acutance ignoring the motion boarder. Sharp motion
        # boarders can potentially get rewarded with high acutance.
        current_acu = compute_acutance(ave_image, dy_max, dx_max)

        if current_acu > best_results['acutance']:
            best_results['acutance'] = current_acu
            best_results['ave_image'] = ave_image
            best_results['ref_image'] = ref_image
            best_results['smooth_sigma'] = current_args['smooth_sigma']
            best_results['smooth_sigma_time'] = \
                current_args['smooth_sigma_time']
        if logger:
            logger(f'\t\tResulting acutance={current_acu:.4f}')
    if logger:
        logger(
            f'Found best motion parameters in {time() - start_time:.0f} '
            f'seconds, with image acutance={best_results["acutance"]:.4f}, '
            f'for parameters: smooth_sigma={best_results["smooth_sigma"]}, '
            f'smooth_sigma_time={best_results["smooth_sigma_time"]}')
    return best_results


def create_ave_image(ref_image: np.ndarray,
                     suite2p_args: dict,
                     batch_size: int = 500) -> dict:
    """Run suite2p image motion correction over a full movie.

    Parameters
    ----------
    ref_image : numpy.ndarray, (N, M)
        Reference image to correlate with movie frames.
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
            frames before correlation. Used as minimum start value in parameter
            search (float).
        ``"smooth_sigma_time"``
            Time Gaussian smoothing of frames to apply before correlation.
            Used as minimum start value in parameter search (float).
    batch_size : int, optional

    """
    with h5py.File(suite2p_args['h5py']) as raw_file:
        frames_dataset = raw_file[suite2p_args['h5py_key']]
        tot_frames = frames_dataset.shape[0]
        ave_frame = np.zeros((frames_dataset.shape[1],
                              frames_dataset.shape[2]))
        dy_max = 0
        dx_max = 0
        for start_idx in np.arange(0, tot_frames, batch_size):
            end_idx = start_idx + batch_size
            if end_idx > tot_frames:
                end_idx = tot_frames
            frames = frames_dataset[start_idx:end_idx]
            add_required_parameters(suite2p_args)
            frames, dy, dx, _, _, _, _ = register_frames(refAndMasks=ref_image,
                                                         frames=frames,
                                                         ops=suite2p_args)
            dy_max = max(dy_max, np.fabs(dy).max())
            dx_max = max(dx_max, np.fabs(dx).max())
            ave_frame += frames.sum(axis=0) / tot_frames

    return {'ave_image': ave_frame,
            'dy_max': int(dy_max),
            'dx_max': int(dx_max)}


def add_required_parameters(suite2p_args: dict):
    """Check that minimum parameters needed by suite2p registration are
    available. If not add them to the suite2p_args dict.

    Parameters
    ----------
    suite2p_args : dict
        Suite2p ops dictionary with potentially missing values.
    """
    if suite2p_args.get('1Preg') is None:
        suite2p_args['1Preg'] = False
    if suite2p_args.get('bidiphase') is None:
        suite2p_args['bidiphase'] = False
    if suite2p_args.get('nonrigid') is None:
        suite2p_args['nonrigid'] = False
    if suite2p_args.get('norm_frames') is None:
        suite2p_args['norm_frames'] = True


def compute_acutance(image: np.ndarray,
                     cut_y: int = 0,
                     cut_x: int = 0) -> float:
    """Compute the acutance (sharpness) of an image.

    Parameters
    ----------
    image : numpy.ndarray, (N, M)
        Image to compute acutance of.
    cut_y : int
        Number of pixels to cut from the begining and end of the y axis.
    cut_x : int
        Number of pixels to cut from the begining and end of the x axis.

    Returns
    -------
    acutance : float
        Acutance of the image.
    """
    if cut_y <= 0 and cut_x <= 0:
        cut_image = image
    elif cut_y > 0 and cut_x <= 0:
        cut_image = image[cut_y:-cut_y, :]
    elif cut_y <= 0 and cut_x > 0:
        cut_image = image[:, cut_x:-cut_x]
    else:
        cut_image = image[cut_y:-cut_y, cut_x:-cut_x]
    grady, gradx = np.gradient(cut_image)
    return (grady ** 2 + grady ** 2).mean()
