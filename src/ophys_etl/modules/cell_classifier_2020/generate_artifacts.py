"""
Script to just generate the artifacts from a set of ROIs and a movie
"""

from typing import Tuple

import numpy as np
import PIL.Image
import h5py
import json

import pathlib

from ophys_etl.types import ExtractROI

import argparse
import time


def scale_to_uint8(data, min_val, max_val):

    data = np.where(data<min_val, min_val, data)
    data = np.where(data>max_val, max_val, data)

    delta = float(max_val-min_val)
    new = np.round(255.0*(data.astype(float)-min_val)/delta).astype(np.uint8)
    assert new.max() <= 255
    assert new.min() >= 0
    return new

def centroid_pixel_from_roi(roi: ExtractROI) -> Tuple[int, int]:
    """
    output pixel will be in full FOV coordinates

    output pixel will be in (row, column) coordinates
    """
    row = 0.0
    col = 0.0
    n = 0.0
    for ir in range(roi['height']):
        for ic in range(roi['width']):
            if not roi['mask'][ir][ic]:
                continue
            row += roi['y']+ir
            col += roi['x']+ic
            n += 1.0

    row = np.round(row/n).astype(int)
    col = np.round(col/n).astype(int)
    return (row, col)


def thumbnail_bounds_from_ROI(
        roi: ExtractROI,
        input_fov_shape: Tuple[int, int],
        output_fov_shape: Tuple[int, int]):
    """
    Get the field of view bounds from an ROI
    Parameters
    ----------
    roi: ExtractROI
    input_fov_shape: Tuple[int, int]
        The 2-D shape of the full field of view
    output_fov_shape: Tuple[int, int]
        The output fov shape

    Returns
    -------
    origin: Tuple[int, int]
        The origin of a sub field of view containing
        the ROI
    shape: Tuple[int, int]
        The shape of the sub field of view
    padding:
        To be passed to np.pad
    Notes
    -----
    Will try to return a square that is a multiple of 16
    on a side (this is what FFMPEG expects)
    """

    (row_center,
     col_center) = centroid_pixel_from_roi(roi)

    half_height = output_fov_shape[0]//2
    half_width = output_fov_shape[1]//2

    rowmin = max(0, row_center-half_height)
    rowmax = min(input_fov_shape[0], row_center+half_height)

    colmin = max(0, col_center-half_width)
    colmax = min(input_fov_shape[1], col_center+half_width)

    is_padded = False
    padding = [[0, 0], [0, 0]]
    if row_center-rowmin < half_height:
        padding[0][0] = half_height-(row_center-rowmin)
        is_padded = True
    if rowmax-row_center < half_height:
        padding[0][1] = half_height-(rowmax-row_center)
        is_padded = True
    if col_center-colmin < half_width:
        padding[1][0] = half_width-(col_center-colmin)
        is_padded = True
    if colmax-col_center < half_width:
        padding[1][1] = half_width-(colmax-col_center)
        is_padded = True
    new_shape = (rowmax-rowmin, colmax-colmin)

    if not is_padded:
        padding = None

    return (rowmin, colmin), new_shape, padding


def get_artifacts(
        roi: ExtractROI,
        max_projection: np.ndarray,
        avg_projection: np.ndarray,
        correlation_projection=None):
    (origin,
     shape,
     padding) = thumbnail_bounds_from_ROI(roi,
                                          max_projection.shape,
                                         (128, 128))

    row0 = origin[0]
    row1 = row0+shape[0]
    col0 = origin[1]
    col1 = col0+shape[1]

    max_thumbnail = max_projection[row0:row1, col0:col1]
    avg_thumbnail = avg_projection[row0:row1, col0:col1]
    if correlation_projection is not None:
        corr_thumbnail = correlation_projection[row0:row1, col0:col1]
    else:
        corr_thumbnail = None

    mask = np.zeros((row1-row0, col1-col0), dtype=np.uint8)
    for irow in range(roi['height']):
        row = irow + roi['y'] - origin[0]
        if row < 0:
            continue
        if row >= (row1-row0):
            continue
        for icol in range(roi['width']):
            col = icol + roi['x'] - origin[1]
            if col < 0:
                continue
            if col >= (col1-col0):
                continue
            if not roi['mask'][irow][icol]:
                continue
            mask[row, col] = 255

    if padding is not None:
        max_thumbnail = np.pad(max_thumbnail, padding)
        avg_thumbnail = np.pad(avg_thumbnail, padding)
        if corr_thumbnail is not None:
            corr_thumbnail = np.pad(corr_thumbnail, padding)
        mask = np.pad(mask, padding)

    assert max_thumbnail.shape == (128, 128)
    assert avg_thumbnail.shape == (128, 128)
    assert mask.shape == (128, 128)

    if corr_thumbnail is not None:
        assert corr_thumbnail.shape == (128, 128)

    return {'max': max_thumbnail,
            'avg': avg_thumbnail,
            'mask': mask,
            'corr': corr_thumbnail}

def run_artifacts(roi_path=None, video_path=None, out_dir=None, n_roi=10):

    roi_path = pathlib.Path(roi_path)
    if not roi_path.is_file():
        raise RuntimeError(f'{roi_path} is not a file')
    video_path = pathlib.Path(video_path)
    if not video_path.is_file():
        raise RuntimeError(f'{video_path} is not a file')
    out_dir = pathlib.Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    if not out_dir.is_dir():
        raise RuntimeError(f'{out_dir} is not a dir')

    with h5py.File(video_path, 'r') as in_file:
        video_data = in_file['data'][()]
        video_max = video_data.max()
        video_min = video_data.min()
    max_projection = np.max(video_data, axis=0)
    avg_projection = np.mean(video_data, axis=0)
    del video_data

    max020, max099 = np.quantile(max_projection, (0.2, 0.99))
    max_projection = scale_to_uint8(max_projection,
                                    min_val=max020,
                                    max_val=max099)

    avg020, avg099 = np.quantile(avg_projection, (0.2, 0.99))
    avg_projection = scale_to_uint8(avg_projection,
                                    min_val=avg020,
                                    max_val=avg099)

    with open(roi_path, 'rb') as in_file:
        raw_rois = json.load(in_file)
        roi_list = []
        for roi in raw_rois:
            if 'valid_roi' in roi:
                roi['valid'] = roi.pop('valid_roi')
            if 'mask_matrix' in roi:
                roi['mask'] = roi.pop('mask_matrix')
            roi_list.append(roi)

    if n_roi > 0:
        roi_list = roi_list[:n_roi]

    for roi in roi_list:
        artifacts = get_artifacts(roi=roi,
                                  max_projection=max_projection,
                                  avg_projection=avg_projection)

        max_img = PIL.Image.fromarray(artifacts['max'])
        max_img.save(out_dir/f"max_{roi['id']}.png")
        avg_img = PIL.Image.fromarray(artifacts['avg'])
        avg_img.save(out_dir/f"avg_{roi['id']}.png")
        mask_img = PIL.Image.fromarray(artifacts['mask'])
        mask_img.save(out_dir/f"mask_{roi['id']}.png")

    return roi_list

if __name__ == "__main__":

    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_path', type=str, default=None)
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--n_roi', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    run_artifacts(roi_path=args.roi_path,
                  video_path=args.video_path,
                  n_roi=args.n_roi,
                  out_dir=args.out_dir)
