import h5py
import numpy as np
from suite2p.registration.register import pick_initial_reference
from suite2p.registration.rigid import (apply_masks, compute_masks, phasecorr,
                                        phasecorr_reference, shift_frame)


def load_initial_frames(file_path: str,
                        h5py_key: str,
                        nimg_init: int) -> np.ndarray:
    """Load a subset of frames from the data specified by file_path.

    Parameters
    ----------
    file_path : str
        Location of the raw 2Photo, HDF5 data to load.
    h5py_key : str
        Name of the dataset to load from the HDF5 file.
    nimg_init : int
        Number of frames to load from the input HDF5 data.

    Returns
    -------
    frames : array-like, (nimg_init, Ly, Lx)
        Selected frames from the input raw data linearly spaced in index of the
        time axis. If nimg_init > nframes, a number of frames equal to nframes
        is returned.
    """
    with h5py.File(file_path, "r") as hdf5_file:
        # Get a set of linear spaced frames to load from disk.
        tot_frames = hdf5_file[h5py_key].shape[0]
        requested_frames = np.linspace(0,
                                       tot_frames,
                                       1 + min(nimg_init, tot_frames),
                                       dtype=int)[:-1]
        frames = hdf5_file[h5py_key][requested_frames]
    return frames


def compute_reference(frames: np.ndarray,
                      niter: int,
                      maxregshift: float,
                      smooth_sigma: float,
                      smooth_sigma_time: float) -> np.ndarray:
    """Computes a set of reference image from the input frames.

    Modified version of Suite2P's compute_reference function with no updated
    of input frames. Picks initial reference then iteratively aligns frames to
    create reference. Does not reproduce Suite2p 1Photo code path.

    Parameters
    ----------
    frames : array-like, (nimg_init, Ly, Lx)
        Subset of frames to create a reference from.
    niter : int
        Number of iterations to perform when creating the reference image.
    maxregshift : float
        Maximum shift allowed as a fraction of the width or the height, which
        ever is longer.
    smooth_sigma : float
        Width of the Gaussian used to smooth the phase correlation between the
        reference and the frame with which it is being registered.
    smooth_sigma_time : float
        Width of the Gaussian used to smooth weight multiple frames by before
        phase correlation.

    Returns
    -------
    refImg : array-like, (Ly, Lx)
        Reference image created from the input data.
    """
    ref_image = pick_initial_reference(frames)

    for idx in range(niter):
        # Compute the number of frames to select in creating the reference
        # image.
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
        isort = np.argsort(-cmax)[1:nmax]

        # Copy the most correlated frames so we don't shift the original data.
        max_corr_frames = np.copy(frames[isort])
        max_corr_xmax = xmax[isort]
        max_corr_ymax = ymax[isort]
        for frame, dy, dx in zip(max_corr_frames,
                                 max_corr_ymax,
                                 max_corr_xmax):
            frame[:] = shift_frame(frame=frame, dy=dy, dx=dx)

        # Reset reference image
        ref_image = max_corr_frames.mean(axis=0).astype(np.int16)
        # Shift reference image to position of mean shifts to remove any bulk
        # displacement.
        ref_image = shift_frame(
            frame=ref_image,
            dy=int(np.round(-max_corr_ymax.mean())),
            dx=int(np.round(-max_corr_xmax.mean()))
        )

    return ref_image
