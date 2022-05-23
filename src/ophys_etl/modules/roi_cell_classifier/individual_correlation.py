import matplotlib.figure as mplt_figure
from matplotlib.backends.backend_pdf import PdfPages
import time

import argparse
import pathlib
import numpy as np
import h5py
import json
import copy
import re
from typing import Tuple
from scipy.spatial.distance import cdist

from ophys_etl.modules.utils.rois import (
    extract_roi_to_ophys_roi)


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
    print(img.min(),img.max())
    return img


def generate_correlation_thumbnail(
        video: np.ndarray,
        roi: dict,
        thumbnail_shape: Tuple[int, int] = (128, 128),
        filter_fraction: float = 0.2,
        area_threshold: float = 0.9) -> np.ndarray:

    mask_key = find_mask_key(roi)

    timesteps = select_timesteps(video, roi, filter_fraction)
    video = video[timesteps, :, :]

    roi_mask = np.zeros(video.shape[1:], dtype=bool)
    roi_mask[roi['y']:roi['y']+roi['height'],
             roi['x']:roi['x']+roi['width']] = np.array(roi[mask_key])

    soma_circle = find_soma_circle(roi, area_threshold)

    bckgd_mask = find_background_pixels(roi, video.shape[1:], soma_circle)

    # select thumbnail field of view;
    # if thumbnail runs off the edge of the video's field of view,
    # adjust the center (rather than padding it)
    r0 = soma_circle['center'][0] - thumbnail_shape[0]//2
    if r0 < 0:
        r0 = 0
    r1 = r0 + thumbnail_shape[0]
    if r1 > video.shape[1]:
        r1 = video.shape[1]
        r0 = r1 - thumbnail_shape[0]
    c0 = soma_circle['center'][1] - thumbnail_shape[1]//2
    if c0 < 0:
        c0 = 0
    c1 = c0 + thumbnail_shape[1]
    if c1 > video.shape[2]:
        c1 = video.shape[2]
        c0 = c1 - thumbnail_shape[1]
    video = video[:, r0:r1, c0:c1]
    roi_mask = roi_mask[r0:r1, c0:c1]
    bckgd_mask = bckgd_mask[r0:r1, c0:c1]

    img = correlate_pixels(
                video,
                roi_mask,
                bckgd_mask)

    return {'img': img,
            'r0': r0, 'r1': r1,
            'c0': c0, 'c1': c1}



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--roi_path', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--n_roi', type=int, default=20)
    args = parser.parse_args()

    video_path = pathlib.Path(args.video_path)
    assert video_path.is_file()
    roi_path = pathlib.Path(args.roi_path)
    assert roi_path.is_file()
    output_path = pathlib.Path(args.output_path)
    assert output_path.parent.is_dir()

    exp_id_pattern = re.compile('[0-9]+')
    baseline_dir = pathlib.Path('images')
    exp_id = exp_id_pattern.findall(roi_path.name)[0]
    baseline_path = baseline_dir / f'{exp_id}_fine_tuned_correlation_img.h5'
    assert baseline_path.is_file()

    with open(roi_path, 'rb') as in_file:
        raw_roi_list = json.load(in_file)

    with h5py.File(video_path, 'r') as in_file:
        video_data = in_file['data'][()]


    roi_list = []
    for roi in raw_roi_list:
        roi_list.append(roi)
        #if roi['y'] <= 128:
        #    roi_list.append(roi)

    if args.n_roi > 0:
        raw_roi_list = roi_list
        roi_list = []
        while len(roi_list) < args.n_roi:
            roi_list.append(raw_roi_list.pop(0))
            roi_list.append(raw_roi_list.pop(-1))

    t0 = time.time()
    ct = 0

    with PdfPages(output_path, 'w') as pdf_handle:

        for roi in roi_list:
            id_key = find_id_key(roi)
            roi_id = roi[id_key]

            #output_path = output_dir / f'{exp_id}_{roi_id}.png'

            thumbnail = generate_correlation_thumbnail(
                          video_data,
                          roi)

            print(thumbnail['r0'], thumbnail['r1'],
                  thumbnail['c0'], thumbnail['c1'],
                  roi['width'], roi['height'])

            with h5py.File(baseline_path, 'r') as in_file:
                baseline_img = in_file['data'][()]
            baseline_img = baseline_img[
                         thumbnail['r0']:thumbnail['r1'],
                         thumbnail['c0']:thumbnail['c1']]

            fig = mplt_figure.Figure(figsize=(20, 20))
            raw_axis = fig.add_subplot(2,2,1)
            baseline_axis = fig.add_subplot(2, 2, 2)

            raw_axis.imshow(baseline_img, cmap='gray')
            raw_this_axis = fig.add_subplot(2, 2, 3)

            raw_this_axis.imshow(add_roi_to_img(
                                    thumbnail['img'],
                                    None,
                                    None,
                                    None,
                                    mn=-1.0,
                                    mx=1.0))
            this_axis = fig.add_subplot(2, 2, 4)
            baseline_axis.imshow(add_roi_to_img(baseline_img,
                                            roi,
                                            thumbnail['r0'],
                                            thumbnail['c0']))
            raw_axis.set_title(f'{exp_id}; ROI {roi_id} baseline',
                                fontsize=15)
            this_axis.imshow(add_roi_to_img(thumbnail['img'],
                                        roi,
                                        thumbnail['r0'],
                                        thumbnail['c0'],
                                        mn=-1.0,
                                        mx=1.0))
        
            title = f"img range: ["
            title += f"{thumbnail['img'].min():.2e}, "
            title += f"{thumbnail['img'].max():.2e}]"
            this_axis.set_title(title, fontsize=15)
            raw_this_axis.set_title('experimental', fontsize=15)
            for a in [raw_axis, baseline_axis, this_axis, raw_this_axis]:
                for ii in range(32, 128, 32):
                    a.axhline(ii, color='r', alpha=0.25)
                    a.axvline(ii, color='r', alpha=0.25)
            fig.tight_layout()
            #fig.savefig(output_path)
            pdf_handle.savefig(fig)
            ct += 1
            duration = time.time()-t0
            per = duration/ct
            print(f'{ct} of {len(roi_list)} in {duration:.2e} seconds -- {per:.2e} per')
    print(f'wrote {output_path}')
