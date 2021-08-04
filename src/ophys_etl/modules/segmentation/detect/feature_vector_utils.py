from typing import Tuple, Optional
from itertools import product
import numpy as np


def choose_timesteps(
            sub_video: np.ndarray,
            seed_pt: Tuple[int, int],
            filter_fraction: float,
            img_data: np.ndarray,
            pixel_ignore: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Choose the timesteps to use for a given seed when calculating Pearson
    correlation coefficients.

    Parameters
    ----------
    sub_video: np.ndarray
        A subset of a video to be correlated.
        Shape is (n_time, n_row, n_cols)

    seed_pt: Tuple[int, int]
        The seed point around which to build the ROI

    filter_fraction: float
        The fraction of brightest timesteps to be used in calculating
        the Pearson correlation between pixels

    img_data: np.ndarray
        The metric image being used for seeding

    pixel_ignore: Optional[np.ndarray]:
        An 1-D array of booleans marked True at any pixels
        that should be ignored, presumably because they have already been
        selected as ROI pixels. (default: None)

    Returns
    -------
    global_mask: np.ndarray
        Array of timesteps (ints) to be used for calculating correlations
    """
    # fraction of timesteps to discard
    discard = 1.0-filter_fraction

    # start assembling mask in timesteps
    trace = sub_video[:, seed_pt[0], seed_pt[1]]
    thresh = np.quantile(trace, discard)
    global_mask = []
    mask = np.where(trace >= thresh)[0]
    global_mask.append(mask)

    image_flat = img_data.flatten()
    if pixel_ignore is not None:
        pixel_ignore_flat = pixel_ignore.flatten()
    else:
        pixel_ignore_flat = np.zeros(len(image_flat), dtype=bool)

    pixel_indexes = np.arange(len(image_flat), dtype=int)
    valid_pixels = np.logical_not(pixel_ignore_flat)
    image_flat = image_flat[valid_pixels]
    pixel_indexes = pixel_indexes[valid_pixels]

    image_max = image_flat.max()
    t25, t75 = np.quantile(image_flat, (0.25, 0.75))
    std = (t75-t25)/1.34896

    for ds in (-1, -2):
        val = image_max + ds*std
        ii = pixel_indexes[np.argmin(np.abs(val-image_flat))]
        pt = np.unravel_index(ii, sub_video.shape[1:])
        trace = sub_video[:, pt[0], pt[1]]
        thresh = np.quantile(trace, discard)
        mask = np.where(trace >= thresh)[0]
        global_mask.append(mask)

    return np.unique(np.concatenate(global_mask))


def select_window_size(
        seed_pt: Tuple[int, int],
        img: np.ndarray,
        target_z_score: float=2.0,
        window_min: int=20,
        window_max: int=300,
        pixel_ignore=None) -> int:

    if pixel_ignore is not None:
        pixel_use = np.logical_not(pixel_ignore)
    else:
        pixel_use = np.ones(img.shape, dtype=bool)

    window = None
    seed_flux = img[seed_pt[0], seed_pt[1]]
    z_score = 0.0
    while z_score < target_z_score:
        if window is None:
            window = window_min
        else:
            window = 3*window//2
        r0 = max(0, seed_pt[0]-window)
        r1 = min(img.shape[0], seed_pt[0]+window+1)
        c0 = max(0, seed_pt[1]-window)
        c1 = min(img.shape[1], seed_pt[1]+window+1)
        local_mask = pixel_use[r0:r1, c0:c1]
        local_img = img[r0:r1, c0:c1]
        background = local_img[local_mask].flatten()
        mu = np.mean(background)
        q25, q75 = np.quantile(background, (0.25, 0.75))
        std = (q75-q25)/1.34896
        z_score = (seed_flux-mu)/std
        if window >= window_max:
            break
    print(f'seed {seed_pt} window {window} -- z {z_score:.2e} -- {seed_flux:.2e} {mu:.2e} {std:.2e}')
    return window
