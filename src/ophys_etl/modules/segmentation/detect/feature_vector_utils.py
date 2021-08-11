from typing import Tuple, Optional
import numpy as np


def correlate_to_single_pixel(
        sub_video: np.ndarray,
        pixel: Tuple[int, int],
        filter_fraction: float = 0.2) -> np.ndarray:
    """
    Calculate the Pearson correlation coefficient of all pixels
    in a video against a single pixel

    Parameters
    -----------
    sub_video: np.ndarray
        (n_time, n_rows, n_cols)

    pixel: Tuple[int, int]

    filter_fraction: float
        default: 0.2

    Returns
    -------
    correlation_coeffs: np.ndarray
        (nrows, ncols)
    """
    pixel_trace = sub_video[:, pixel[0], pixel[1]]
    th = np.quantile(pixel_trace, 1.0-filter_fraction)
    valid_timesteps = (pixel_trace >= th)
    sub_video = sub_video[valid_timesteps, :, :]
    img_shape = sub_video.shape[1:]
    i_pixel = np.ravel_multi_index(pixel, img_shape)
    sub_video = sub_video.reshape(sub_video.shape[0], -1)
    mean_image = np.mean(sub_video, axis=0)
    sub_video = sub_video - mean_image
    var = np.mean(sub_video**2, axis=0)
    sub_video = sub_video.transpose()
    numerator = np.dot(sub_video, sub_video[i_pixel, :])/sub_video.shape[1]
    corr = numerator/np.sqrt(var*var[i_pixel])
    return corr.reshape(img_shape)


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
        An 1-D array of booleans marked True at any pixels
        that should be ignored, presumably because they have already been
        selected as ROI pixels. (default: None)

    Returns
    -------
    global_mask: np.ndarray
        Array of timesteps (ints) to be used for calculating correlations
    """
    if pixel_ignore is not None:
        if pixel_ignore.shape != image_data.shape:
            msg = f"pixel_ignore.shape {pixel_ignore.shape}\n"
            msg += f"image_data.shape {image_data.shape}\n"
            msg += "These should be the same"
            raise RuntimeError(msg)

    # fraction of timesteps to discard
    discard = 1.0-filter_fraction

    # start assembling mask in timesteps
    trace = sub_video[:, seed_pt[0], seed_pt[1]]
    thresh = np.quantile(trace, discard)
    global_mask = []
    mask = np.where(trace >= thresh)[0]
    global_mask.append(mask)

    pearson_img = correlate_to_single_pixel(
                      sub_video,
                      seed_pt)

    pearson_img = pearson_img.flatten()

    # now select the pixels that are closest to
    # image_data.max() - sigma and
    # image_data.max() - 2*sigma

    if pixel_ignore is not None:
        pixel_ignore_flat = pixel_ignore.flatten()
    else:
        pixel_ignore_flat = np.zeros(len(pearson_img), dtype=bool)

    pixel_indices = np.arange(len(pearson_img), dtype=int)
    valid_pixels = np.logical_not(pixel_ignore_flat)
    pearson_img = pearson_img[valid_pixels]
    pixel_indices = pixel_indices[valid_pixels]

    pearson_max = pearson_img.max()
    pearson_min = pearson_img.min()
    t25, t75 = np.quantile(pearson_img, (0.25, 0.75))
    std = (t75-t25)/1.34896
    med_val = np.median(pearson_img)

    for val in (pearson_max-std,
                med_val,
                pearson_min+std,
                pearson_min):
        ii = pixel_indices[np.argmin(np.abs(val-pearson_img))]
        pt = np.unravel_index(ii, sub_video.shape[1:])
        trace = sub_video[:, pt[0], pt[1]]
        thresh = np.quantile(trace, discard)
        mask = np.where(trace >= thresh)[0]
        global_mask.append(mask)

    return np.unique(np.concatenate(global_mask))


def select_window_size(
        seed_pt: Tuple[int, int],
        image_data: np.ndarray,
        target_z_score: float = 2.0,
        window_min: int = 20,
        window_max: int = 300,
        pixel_ignore: Optional[np.ndarray] = None) -> int:
    """
    For an image, find the window half side length
    centered on a pixel such that the specified pixel
    is at least N-sigma brighter than the distribution
    of pixels the image.

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
    It grows the window by a factor for 11/10 at each step.
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
        q25, q75 = np.quantile(background, (0.25, 0.75))
        std = max(1.0e-10, (q75-q25)/1.34896)
        z_score = (seed_flux-mu)/std
        if window >= window_max:
            break
        if r0 == 0 and r1 == image_data.shape[0]:
            if c0 == 0 and c1 == image_data.shape[1]:
                break
    return window
