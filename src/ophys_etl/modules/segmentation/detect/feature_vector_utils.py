from typing import Tuple, Optional, List
import numpy as np
from ophys_etl.modules.segmentation.utils.stats_utils import (
    estimate_std_from_interquartile_range)


def choose_extreme_pixels(
        image_data: np.ndarray,
        delta_sigma: List[float],
        pixel_ignore: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """
    For a specified list of values delta_sigma, choose the pixels in an
    image that are that many standard deviations from the maximum
    and minimum flux values

    i.e. if delta_sigma = [1, 2]

    The output will be the row, column coordinates of the pixels
    whose flux values are closest to

    [flux_min + 2*sigma,
     flux_min + 1*sigma,
     flux_max - 2*sigma,
     flux_max - 1*sigma]

    Note: for each delta_sigma value, only one pixel will be found
    using np.argmin(flux-target_flux)

    Parameters
    ----------
    image_data: np.ndarray
        Image data used for determining flux of pixels

    delta_sigma: List[float]
        List of distances (in units of the standard deviation) from
        extremities to find

    pixel_ignore: Optional[np.ndarray]:
        A boolean mask marked True at any pixels
        that should be ignored, presumably because
        they have already been selected as ROI pixels.
        (default: None)

    Returns
    -------
    interesting_points: List[Tupple[int, int]]
        Coordinates of interesting points in row, column space.
        Points will be in ascending flux order.

    """
    if pixel_ignore is not None:
        if pixel_ignore.shape != image_data.shape:
            msg = f"pixel_ignore.shape {pixel_ignore.shape}\n"
            msg += f"image_data.shape {image_data.shape}\n"
            msg += "These should be the same"
            raise RuntimeError(msg)

    image_flat = image_data.flatten()
    if pixel_ignore is not None:
        pixel_ignore_flat = pixel_ignore.flatten()
    else:
        pixel_ignore_flat = np.zeros(len(image_flat), dtype=bool)

    pixel_indices = np.arange(len(image_flat), dtype=int)
    valid_pixels = np.logical_not(pixel_ignore_flat)
    image_flat = image_flat[valid_pixels]
    pixel_indices = pixel_indices[valid_pixels]

    image_max = image_flat.max()
    image_min = image_flat.min()
    std = estimate_std_from_interquartile_range(image_flat)

    flux_values = []
    for ds in delta_sigma:
        flux_values.append(image_max-ds*std)
        flux_values.append(image_min+ds*std)
    flux_values.sort()

    interesting_points = []
    for flux in flux_values:
        ii = pixel_indices[np.argmin(np.abs(flux-image_flat))]
        pt = np.unravel_index(ii, image_data.shape)
        interesting_points.append(tuple(pt))
    return interesting_points


def choose_timesteps(
            sub_video: np.ndarray,
            seed_pt: Tuple[int, int],
            filter_fraction: float,
            image_data: np.ndarray,
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

    image_data: np.ndarray
        The metric image being used for seeding

    pixel_ignore: Optional[np.ndarray]:
        A boolean mask marked True at any pixels
        that should be ignored, presumably because
        they have already been selected as ROI pixels.
        (default: None)

    Returns
    -------
    global_time_mask: np.ndarray
        Array of timesteps (ints) to be used for calculating correlations
    """

    # fraction of timesteps to discard
    discard = 1.0-filter_fraction

    # start assembling mask in timesteps
    trace = sub_video[:, seed_pt[0], seed_pt[1]]
    thresh = np.quantile(trace, discard)
    global_time_mask = []
    mask = np.where(trace >= thresh)[0]
    global_time_mask.append(mask)

    # now select the pixels that are closest to
    #
    # image_data.max() - sigma and
    # image_data.min() + sigma
    #
    # (ignoring pixels masked by pixel_ignore)
    #
    # so that we get timesteps that reasonably
    # span the dynamic range of the ROI and its
    # background

    interesting_points = choose_extreme_pixels(
                             image_data,
                             [1.0],
                             pixel_ignore=pixel_ignore)

    print('\ninteresting points')

    for pt in interesting_points:
        print(f'{pt}')
        trace = sub_video[:, pt[0], pt[1]]
        thresh = np.quantile(trace, discard)
        mask = np.where(trace >= thresh)[0]
        global_time_mask.append(mask)

    print('')

    return np.unique(np.concatenate(global_time_mask)), interesting_points


def select_window_size(
        seed_pt: Tuple[int, int],
        image_data: np.ndarray,
        target_z_score: float = 2.0,
        window_min: int = 20,
        window_max: int = 300,
        pixel_ignore: Optional[np.ndarray] = None) -> int:
    """
    For an image, find the window half side length
    centered on a pixel with (row, column) coordinates
    seed_pt such that the specified pixel is at least
    N-sigma brighter than the distribution of pixels the
    image.

    Parameters
    ----------
    seed_pt: Tuple[int, int]
        (row, col) of the specified point

    image_data: np.ndarray

    target_z_score: float
        The target z-score of seed_pt within image
        (default=2.0)

    window_min: int
        Minimum window size to return (default=20)

    window_max: int
        Maximum window size to return. If the window size
        exceeds this limit without matching or exceeding
        target_z_score, return the last window size tested.
        (default=300)

    pixel_ignore: Optional[np.ndarray]
        A mask marked as True for pixels that are to be
        ignored when calculating the z-score of seed_pt

    Returns
    -------
    window: int
        The half side length (in pixels) of a window
        centered on seed_pt such that seed_pt is at
        least target_z_score standard deviations
        brighter than all of the pixels in the window

    Notes
    -----
    This method starts with a window size of window_min.
    It grows the window by 10% at each step.
    It returns the first value of window that either
    exceeds window_max or results in a z-score exceeding
    target_z_score.
    """
    if pixel_ignore is not None:
        if pixel_ignore.shape != image_data.shape:
            msg = f"pixel_ignore.shape {pixel_ignore.shape}\n"
            msg += f"image_data.shape {image_data.shape}\n"
            msg += "These should be the same"
            raise RuntimeError(msg)
        pixel_use = np.logical_not(pixel_ignore)
    else:
        pixel_use = np.ones(image_data.shape, dtype=bool)

    window = None
    seed_flux = image_data[seed_pt[0], seed_pt[1]]
    z_score = 0.0
    while z_score < target_z_score:
        if window is None:
            window = window_min
        else:
            window += max(1, window//10)
        r0 = max(0, seed_pt[0]-window)
        r1 = min(image_data.shape[0], seed_pt[0]+window+1)
        c0 = max(0, seed_pt[1]-window)
        c1 = min(image_data.shape[1], seed_pt[1]+window+1)
        local_mask = pixel_use[r0:r1, c0:c1]
        local_image = image_data[r0:r1, c0:c1]
        background = local_image[local_mask].flatten()
        mu = np.mean(background)
        std = estimate_std_from_interquartile_range(background)
        std = max(1.0e-10, std)

        z_score = (seed_flux-mu)/std
        if window >= window_max:
            break
        if r0 == 0 and r1 == image_data.shape[0]:
            if c0 == 0 and c1 == image_data.shape[1]:
                break
    return window
