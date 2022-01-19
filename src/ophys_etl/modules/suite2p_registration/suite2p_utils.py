import h5py
import numpy as np
from suite2p.registration.register import pick_initial_reference
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
        # Get a set of linear spaced frames to load from disk.
        tot_frames = hdf5_file[h5py_key]
        requested_frames = np.linspace(0,
                                       tot_frames,
                                       1 + min(n_frames, tot_frames),
                                       dtype=int)[:-1]
        frames = hdf5_file[h5py_key][requested_frames]
    return frames


def compute_reference(frames: np.ndarray,
                      output_prefix: str,
                      maxregshift: float,
                      smooth_sigma: float,
                      smooth_sigma_time: float,
                      niter: int = 25,
                      rtol: float = 1e-4,
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
    maxregshift : float
        Maximum shift allowed as a fraction of the image width or height, which
        ever is longer.
    smooth_sigma : float
        Width of the Gaussian used to smooth the phase correlation between the
        reference and the frame with which it is being registered.
    smooth_sigma_time : float
        Width of the Gaussian used to smooth between multiple frames by before
        phase correlation.
    niter : int
        Max number of iterations to perform when creating the reference image.
        Exit if niter is reached.
    rtol : float
        Relative tolerance to declare convergence.
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

    previous_value = 1e-16
    for idx in range(niter):
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
        # Quality cut on cmax. Currently we use the top half of the input
        # frames based on correlation amplitude.
        cmax_mask = cmax > cmax.median()
        print(f"Using {cmax_mask.sum()} frames...")

        # Copy the most correlated frames so we don't shift the original data.
        max_corr_frames = np.copy(frames[cmax_mask])
        max_corr_ymax = ymax[cmax_mask]
        max_corr_xmax = xmax[cmax_mask]
        max_corr_cmax = cmax[cmax_mask]
        # Apply shift to the copy of the frames.
        for frame, dy, dx in zip(max_corr_frames,
                                 max_corr_ymax,
                                 max_corr_xmax):
            frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)

        # Create a new reference image from the weighted average of the most
        # correlated frames weighted by their correlation^2.
        ref_image = np.average(max_corr_frames,
                               weights=max_corr_cmax ** 2,
                               axis=0).astype(np.int16)
        # Shift reference image to position of mean shifts to remove any bulk
        # displacement.
        ref_image = shift_frame(
            frame=ref_image,
            dy=int(np.round(-np.average(max_corr_ymax,
                                        weights=max_corr_cmax ** 2))),
            dx=int(np.round(-np.average(max_corr_xmax,
                                        weights=max_corr_cmax ** 2)))
        )

        # Board cut in case there are large shifts.
        min_y_shift = int(np.fabs(max_corr_ymax).max())
        min_x_shift = int(np.fabs(max_corr_xmax).max())
        max_y_shift = -min_y_shift if min_y_shift > 0 else -1
        max_x_shift = -min_y_shift if min_y_shift > 0 else -1

        # Compute our stopping criteria.
        # Compute the gradient over our image outside of the motion boarder.
        # We use a gradient over 2 pixels to get slightly more stable results.
        grady, gradx = np.gradient(ref_image, 2)
        grad_magnitude = grady ** 2 + gradx ** 2
        acutance = grad_magnitude[min_y_shift:max_y_shift,
                                  min_x_shift:max_x_shift].mean()

        # Potential other option for convergence is the variance of the image.
        var= ref_image[max_y_shift:max_y_shift,
                       max_x_shift:max_x_shift].var(ddof=1)

        # Converge on the acutance of the non-motion boarder cut image.
        current_quality = acutance
        if abs(current_quality - previous_value) / previous_value < rtol:
            break
        else:
            previous_value = current_quality
    print(f"Created reference image in {idx + 1} iterations...")
    return ref_image


def compute_reference_old(frames: np.ndarray,
                          output_prefix: str,
                          maxregshift: float,
                          smooth_sigma: float,
                          smooth_sigma_time: float,
                          niter: int = 8,
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

    grady, gradx = np.gradient(ref_image)
    image_data = pd.DataFrame(
        data={"start_image": ref_image.flatten(),
              "start_gradient": (grady ** 2 + gradx ** 2).flatten()})
    corr_data = pd.DataFrame()

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
        corr_data[f"ymax{idx}"] = ymax
        corr_data[f"xmax{idx}"] = xmax
        corr_data[f"cmax{idx}"] = cmax

        # Find the indexes of the frames that are the most correlated and
        # select the first nmax.
        isort = np.argsort(-cmax)[1:nmax]

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

        # Compute our stopping criteria.
        max_y_shift = int(np.fabs(max_corr_ymax).max())
        max_x_shift = int(np.fabs(max_corr_xmax).max())
        grady, gradx = np.gradient(ref_image, 2)
        grad_magnitude = grady ** 2 + gradx ** 2
        image_data[f"ref_image{idx}"] = ref_image.flatten()
        image_data[f"grad2_ref{idx}"] = grad_magnitude.flatten()

        accutance = grad_magnitude[max_y_shift:-max_y_shift,
                                   max_x_shift:-max_x_shift].mean()
        variance_sum = (
            (ref_image[max_y_shift:-max_y_shift, max_x_shift:-max_x_shift]
             - np.mean(ref_image[max_y_shift:-max_y_shift,
                                 max_x_shift:-max_x_shift])) ** 2).sum()

    return ref_image
