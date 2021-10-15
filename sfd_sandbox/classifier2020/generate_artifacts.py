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
    Notes
    -----
    Will try to return a square that is a multiple of 16
    on a side (this is what FFMPEG expects)
    """

    # center the thumbnail on the ROI
    row_center = int(roi['y'] + roi['height']//2)
    col_center = int(roi['x'] + roi['width']//2)

    rowmin = max(0, row_center - output_fov_shape[0]//2)
    rowmax = rowmin + output_fov_shape[0]
    colmin = max(0, col_center - output_fov_shape[1]//2)
    colmax = colmin + output_fov_shape[1]

    if rowmax >= input_fov_shape[0]:
        rowmin = max(0, input_fov_shape[0]-output_fov_shape[0])
        rowmax = min(input_fov_shape[0], rowmin+output_fov_shape[0])
    if colmax >= input_fov_shape[1]:
        colmin = max(0, input_fov_shape[1]-output_fov_shape[1])
        colmax = min(input_fov_shape[1], colmin+output_fov_shape[1])

    new_shape = (rowmax-rowmin, colmax-colmin)
    if new_shape != output_fov_shape:
        raise RuntimeError(f'wanted shape {output_fov_shape}\n'
                           f'got {new_shape}')

    return (rowmin, colmin), new_shape


def get_artifacts(roi, max_projection, avg_projection):
    (origin,
     shape) = thumbnail_bounds_from_ROI(roi,
                                        max_projection.shape,
                                        (128, 128))

    row0 = origin[0]
    row1 = row0+shape[0]
    col0 = origin[1]
    col1 = col0+shape[1]

    max_thumbnail = max_projection[row0:row1, col0:col1]
    avg_thumbnail = avg_projection[row0:row1, col0:col1]

    mask = np.zeros(shape, dtype=np.uint8)
    for irow in range(roi['height']):
        row = irow + roi['y'] - origin[0]
        if row < 0:
            continue
        if row >= shape[0]:
            continue
        for icol in range(roi['width']):
            col = icol + roi['x'] - origin[1]
            if col < 0:
                continue
            if col >= shape[1]:
                continue
            if not roi['mask'][irow][icol]:
                continue
            mask[row, col] = 255

    return {'max': max_thumbnail,
            'avg': avg_thumbnail,
            'mask': mask}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--roi_path', type=str, default=None)
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--n_roi', type=int, default=10)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    roi_path = pathlib.Path(args.roi_path)
    if not roi_path.is_file():
        raise RuntimeError(f'{roi_path} is not a file')
    video_path = pathlib.Path(args.video_path)
    if not video_path.is_file():
        raise RuntimeError(f'{video_path} is not a file')
    out_dir = pathlib.Path(args.out_dir)
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

    if args.n_roi > 0:
        roi_list = roi_list[:args.n_roi]

    t0 = time.time()
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
    print(f'that took {time.time()-t0:.2e}')
