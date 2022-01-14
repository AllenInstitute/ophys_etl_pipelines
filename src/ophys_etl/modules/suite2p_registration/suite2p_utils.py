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
        tot_frames = hdf5_file[h5py_key].shape[0]
        requested_frames = np.linspace(0,
                                       tot_frames,
                                       1 + min(n_frames, tot_frames),
                                       dtype=int)[:-1]
        frames = hdf5_file[h5py_key][requested_frames]
    return frames


def compute_reference(frames: np.ndarray,
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
        Number of iterations to perform when creating the reference image.
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
        cmax_mean = cmax.mean()
        cmax_std = cmax.std(ddof=1)
        cmax_mask = cmax > np.percentile(cmax, 16)

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

        # Compute our stopping criteria.
        max_y_shift = int(np.fabs(max_corr_ymax).max())
        max_x_shift = int(np.fabs(max_corr_xmax).max())
        # Compute the gradient over our image outside of the motion boarder.
        grady, gradx = np.gradient(
            ref_image[max_y_shift:-max_y_shift, max_x_shift:-max_x_shift])
        grad_magnitude = np.sqrt(grady ** 2 + gradx ** 2)
        # Current convergence criteria: Difference between the median and the
        # 95th percentile.
        percentiles = np.percentile(grad_magnitude, [16, 50, 84, 95])
        current_quality = 2 * (percentiles[3] - percentiles[1]) / (percentiles[2] - percentiles[0])
        print("percentiles [16, 50, 84, 95]:", percentiles)
        print("current quality", current_quality)
        print("cmax percentiles:", np.percentile(cmax, [5, 25, 50, 75, 95]))
        if abs(current_quality - previous_value) / previous_value < rtol:
            break
        else:
            previous_value = current_quality
    print(f"Created reference image in {idx + 1} iterations...")
    return ref_image
