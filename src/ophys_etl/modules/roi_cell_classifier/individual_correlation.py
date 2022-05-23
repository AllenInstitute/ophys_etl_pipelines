from typing import List, Tuple, Any
import time

import argparse
import pathlib
import numpy as np
import h5py
import json
import copy
import re
import multiprocessing
import PIL.Image
from scipy.spatial.distance import cdist

from ophys_etl.utils.rois import (
    extract_roi_to_ophys_roi,
    clip_roi)

from ophys_etl.types import ExtractROI

from ophys_etl.utils.array_utils import (
    normalize_array,
    pairwise_distances)

from ophys_etl.utils.multiprocessing_utils import (
        _winnow_process_list)


def find_mask_key(roi: dict):
    if 'mask' in roi:
        mask_key = 'mask'
    elif 'mask_matrix' in roi:
        mask_key = 'mask_matrix'
    else:
        raise RuntimeError("No obvious mask key in ROI")

    return mask_key


def find_id_key(roi: dict):
    if 'id' in roi:
        return 'id'
    elif 'roi_id' in roi:
        return 'roi_id'
    raise RuntimeError("No obvious ID key in ROI")


def find_best_circle(
        pixel_mask: np.ndarray,
        area_threshold: float):

    n_pixels = pixel_mask.shape[0]*pixel_mask.shape[1]

    flat_mask = pixel_mask.flatten()
    n_by_n_mask = np.array([flat_mask]*n_pixels)
    assert n_by_n_mask.shape == (n_pixels, n_pixels)
    del flat_mask

    (pixel_rows,
     pixel_cols) = np.meshgrid(np.arange(pixel_mask.shape[0]),
                               np.arange(pixel_mask.shape[1]),
                               indexing='ij')

    pixel_rows = pixel_rows.flatten()
    pixel_cols = pixel_cols.flatten()
    coords = np.array([pixel_rows, pixel_cols]).transpose()
    pixel_distances = pairwise_distances(coords)
    del coords

    # for each row, only keep the distances to valid pixels
    pixel_distances = pixel_distances[n_by_n_mask].reshape(n_pixels, -1)
    del n_by_n_mask
    assert pixel_distances.shape == (n_pixels, pixel_mask.sum())

    rmax = 0.5*(max(pixel_mask.shape[0], pixel_mask.shape[1])+1.0)
    r_array = np.arange(1.0, rmax, 0.5)
    area_array = np.pi*r_array**2

    r_at_threshold = np.zeros(n_pixels, dtype=float)
    area_at_threshold = np.zeros(n_pixels, dtype=float)

    for radius, area in zip(r_array, area_array):
        valid = (pixel_distances <= radius)
        valid = valid.sum(axis=1).astype(float)
        fraction = valid/area
        flagging = fraction >= area_threshold
        r_at_threshold[flagging] = radius
        area_at_threshold[flagging] = valid[flagging]

    max_dex = np.argmax(r_at_threshold)
    center = (pixel_rows[max_dex], pixel_cols[max_dex])
    return {'center': center,
            'radius': r_at_threshold[max_dex],
            'area': area_at_threshold[max_dex]}


def select_timesteps(
        video: np.ndarray,
        roi: dict,
        filter_fraction: float = 0.2) -> np.ndarray:
    """
    Select the brightest filter_fraction of timesteps from
    the average timeseries of all ROI pixels.

    Parameters:
    -----------
    video: np.ndarray
        The full (ntime, nrows, ncols) video
    roi: dict
        The ROI. The mask of the ROI is specified by 'x', 'y', and 'mask'
        (or 'mask_matrix')
    filter_fraction: float
        Fraction of brightest timesteps to use

    Returns
    -------
    timesteps: np.ndarray
        1D array of ints denoting the selected timesteps
    """

    mask_key = find_mask_key(roi)

    mask = np.zeros(video.shape[1:], dtype=bool)
    mask[roi['y']:roi['y']+roi['height'],
         roi['x']:roi['x']+roi['width']] = np.array(roi[mask_key])

    time_series = np.sum(video[:, mask], axis=1)
    cutoff = np.quantile(time_series, 1.0-filter_fraction)
    return np.where(time_series >= cutoff)[0]


def find_centroid_circle(
        pixel_mask: np.ndarray,
        area_threshold: float):
    """
    pixel_mask is a np.ndarray of booleans
    area_threshold is the fraction of the best fit circle which must be filled

    Returns a dict denoting the circle like
    {'center': (row, col),
     'radius': float}

    The center is taken as the centroid of pixel_mask

    This method is meant to be used on ROIs whose mask is absurdly large such
    that rigorously searching for the best fit circle will fail
    """

    (pixel_rows,
     pixel_cols) = np.meshgrid(np.arange(pixel_mask.shape[0]),
                               np.arange(pixel_mask.shape[1]),
                               indexing='ij')

    pixel_rows = pixel_rows.flatten()
    pixel_cols = pixel_cols.flatten()


    flat_mask = pixel_mask.flatten()
    mean_row = np.round(np.mean(pixel_rows[flat_mask]))
    mean_col = np.round(np.mean(pixel_cols[flat_mask]))

    distances = np.sqrt((pixel_rows-mean_row)**2
                        + (pixel_cols-mean_col)**2)

    r_array = np.unique(distances)
    best_radius = r_array.max()
    for rr in r_array:
        if rr < 1.0e-6:
            continue
        area = np.pi*rr**2
        filled = np.sum(distances < rr)
        if (filled/area) < area_threshold:
            best_radius = rr
            break
    return {'center': (int(mean_row), int(mean_col)),
            'radius': best_radius}


def find_soma_circle(
        roi: dict,
        area_threshold: float) -> dict:
    """
    The returned dict has {'center': (row, col), 'radius': float}
    of the approximate centroid circle
    """

    mask_key = find_mask_key(roi)

    mask = np.array(roi[mask_key])

    if mask.size > 10000:
        circle = find_centroid_circle(mask, area_threshold)
    else:
        circle = find_best_circle(mask, area_threshold)

    row = circle['center'][0]
    col = circle['center'][1]
    return {'center': (row+roi['y'], col+roi['x']),
            'radius': circle['radius']}


def find_background_pixels(
        roi: dict,
        fov_shape: Tuple[int, int],
        roi_circle: dict) -> np.ndarray:
    """
    Find a boolean mask containing an anulus of
    background pixels. There must be as many background pixels
    as ROI pixels.
    """
    roi_mask_key = find_mask_key(roi)
    roi_mask = np.zeros(fov_shape, dtype=bool)
    roi_mask[roi['y']:roi['y']+roi['height'],
             roi['x']:roi['x']+roi['width']] = np.array(roi[roi_mask_key])

    # because we do not want ROI pixels to be included in the background
    roi_mask = np.logical_not(roi_mask)

    (bckgd_rows,
     bckgd_cols) = np.meshgrid(np.arange(fov_shape[0]),
                               np.arange(fov_shape[1]),
                               indexing='ij')

    bckgd_distances = np.sqrt((bckgd_rows-roi_circle['center'][0])**2
                             + (bckgd_cols-roi_circle['center'][1])**2)

    n_roi_pixels = np.sum(roi[roi_mask_key])
    r_array = np.unique(bckgd_distances.flatten())

    for rr in r_array:
        if rr <= roi_circle['radius']:
            continue
        bckgd_mask = (bckgd_distances <= rr)
        bckgd_mask = np.logical_and(bckgd_mask, roi_mask)
        n_bckgd_pixels = np.sum(bckgd_mask)
        if n_bckgd_pixels >= n_roi_pixels:
            break
    return bckgd_mask


def correlate_pixels(
        video: np.ndarray,
        roi_mask: np.ndarray,
        background_mask: np.ndarray) -> np.ndarray:

    video_shape = video.shape

    # cast arrays into (ntime, npixels) arrays

    roi_timeseries = video[:, roi_mask].reshape(video_shape[0], -1).transpose()

    bckgd_timeseries = video[:, background_mask].reshape(video_shape[0], -1)
    bckgd_timeseries = bckgd_timeseries.transpose()

    video = video.reshape(video_shape[0], -1).transpose()

    # find the mean of each pixel's correlation with ROI and
    # background pixels

    correlation_to_roi = 1.0 - cdist(video,
                                     roi_timeseries,
                                     metric='correlation')

    assert correlation_to_roi.shape == (video.shape[0],
                                        roi_timeseries.shape[0])

    correlation_to_roi = np.mean(correlation_to_roi, axis=1)

    correlation_to_bckgd = 1.0 - cdist(video,
                                       bckgd_timeseries,
                                       metric='correlation')

    assert correlation_to_bckgd.shape == (video.shape[0],
                                          bckgd_timeseries.shape[0])

    correlation_to_bckgd = np.mean(correlation_to_bckgd, axis=1)

    # construct image out of the difference between the mean
    # correlation against ROI and the mean correlation
    # background
    img = correlation_to_roi - correlation_to_bckgd
    img = img.reshape(video_shape[1:])
    return img


def generate_individual_correlation_thumbnail(
        video: np.ndarray,
        roi: dict,
        row_indices: Tuple[int, int],
        col_indices: Tuple[int, int],
        filter_fraction: float = 0.2,
        area_threshold: float = 0.9,) -> np.ndarray:

    mask_key = find_mask_key(roi)

    timesteps = select_timesteps(video, roi, filter_fraction)
    video = video[timesteps, :, :]

    roi_mask = np.zeros(video.shape[1:], dtype=bool)
    roi_mask[roi['y']:roi['y']+roi['height'],
             roi['x']:roi['x']+roi['width']] = np.array(roi[mask_key])

    soma_circle = find_soma_circle(roi, area_threshold)

    bckgd_mask = find_background_pixels(roi, video.shape[1:], soma_circle)
    r0 = row_indices[0]
    r1 = row_indices[1]
    c0 = col_indices[0]
    c1 = col_indices[1]

    video = video[:, r0:r1, c0:c1]
    roi_mask = roi_mask[r0:r1, c0:c1]
    bckgd_mask = bckgd_mask[r0:r1, c0:c1]

    img = correlate_pixels(
                video,
                roi_mask,
                bckgd_mask)

    # cutoffs should be -2, 2 since the image
    # is a difference between two mean correlations
    img = normalize_array(
            array=img,
            lower_cutoff=-1.0,
            upper_cutoff=1.0)

    return img


def _individual_correlation_worker(
        video: np.ndarray,
        roi: ExtractROI,
        row_pad: Tuple[int, int],
        col_pad: Tuple[int, int],
        output_dir: Any,
        exp_id: int,
        output_lock: Any):

    img = generate_individual_correlation_thumbnail(
        video=video,
        roi=roi,
        row_indices=(0, video.shape[1]),
        col_indices=(0, video.shape[2]),
        filter_fraction=0.2,
        area_threshold=0.9)

    padding = (row_pad, col_pad)
    img = np.pad(img,
                 pad_width=padding,
                 mode="constant",
                 constant_values=0)

    name = f"self_correlation_{exp_id}_{roi['id']}.png"
    output_path = pathlib.Path(output_dir) / name
    img = PIL.Image.fromarray(img)
    with output_lock:
        img.save(output_path)


def generate_individual_correlation_batch(
        video: np.ndarray,
        roi_list: List[ExtractROI],
        shape_lookup: dict,
        n_workers: int,
        output_dir: Any,
        exp_id: int,
        logger: Any):
    output = []

    mgr = multiprocessing.Manager()
    output_lock = mgr.Lock()
    full_fov_shape = video.shape[1:]

    t0 = time.time()

    logger.info("Starting to create self "
                "correlation thumbnails")
    n_total = len(roi_list)
    n_logged = 0

    process_list = []
    for i_roi, this_roi in enumerate(roi_list):
        this_shape = shape_lookup[this_roi['id']]
        r0 = this_shape['row_indices'][0]
        r1 = this_shape['row_indices'][1]
        c0 = this_shape['col_indices'][0]
        c1 = this_shape['col_indices'][1]
        clipped_video = video[:, r0:r1, c0:c1]
        clipped_roi = clip_roi(
                        roi=this_roi,
                        full_fov_shape=full_fov_shape,
                        row_bounds=this_shape['row_indices'],
                        col_bounds=this_shape['col_indices'])

        process = multiprocessing.Process(
                    target=_individual_correlation_worker,
                    kwargs={
                        'video': clipped_video,
                        'roi': clipped_roi,
                        'row_pad': this_shape['row_pad'],
                        'col_pad': this_shape['col_pad'],
                        'output_dir': output_dir,
                        'exp_id': exp_id,
                        'output_lock': output_lock})
        process.start()
        process_list.append(process)
        while len(process_list) >= n_workers:
            process_list = _winnow_process_list(process_list)
        if i_roi %10 == 0 and i_roi > 0:
            duration = time.time()-t0
            n_found = i_roi
            n_logged = n_found
            per = duration/n_found
            pred = per*n_total
            remaining = pred-duration
            logger.info(f"computed {n_found} of {n_total} "
                        f"in {duration:.2e} seconds; "
                        f"estimate {remaining:.2e} seconds left")

    for process in process_list:
        process.join()

    logger.info("Done with self correlation thumbnails")
