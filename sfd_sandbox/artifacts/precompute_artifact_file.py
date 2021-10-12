import h5py
import json
import pathlib
import numpy as np

from ophys_etl.modules.segmentation.utils.roi_utils import (
    serialize_extract_roi_list,
    extract_roi_to_ophys_roi)

from ophys_etl.modules.segmentation.qc_utils.video_utils import (
    read_and_scale,
    upscale_video_frame)

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
    get_roi_color_map)

import numpy as np

import argparse
import time


def get_traces(video_path, ophys_roi_list):
    with h5py.File(video_path, 'r') as in_file:
        video_data = in_file['data'][()]

    t0 = time.time()
    ct = 0
    trace_lookup = {}
    for roi in ophys_roi_list:
        assert roi.global_pixel_array.shape[1] == 2
        rows = roi.global_pixel_array[:, 0]
        cols = roi.global_pixel_array[:, 1]
        trace = video_data[:, rows, cols]
        trace = np.mean(trace, axis=1)
        assert trace.shape == (video_data.shape[0],)
        trace_lookup[roi.roi_id] = trace
        ct += 1
        if ct % 100 == 0:
            d = time.time()-t0
            p = d/ct
            pred = p*len(ophys_roi_list)
            r = pred-d
            print(f'{ct} in {d:.2e} -- {r:.2e} left of {pred:.2e}')
    return trace_lookup


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--roi_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default='output.h5')
    args = parser.parse_args()

    with open(args.roi_path, 'rb') as in_file:
        raw_rois = json.load(in_file)
    extract_roi_list = []
    ophys_roi_list = []
    for roi in raw_rois:
        roi['valid'] = roi.pop('valid_roi')
        roi['mask'] = roi.pop('mask_matrix')

        extract_roi_list.append(roi)
        ophys_roi_list.append(extract_roi_to_ophys_roi(roi))

    with h5py.File(args.out_path, 'w') as out_file:
        out_file.create_dataset('rois',
                    data=serialize_extract_roi_list(extract_roi_list))

    print('wrote ROIs')

    color_map = get_roi_color_map(ophys_roi_list)
    with h5py.File(args.out_path, 'a') as out_file:
        out_file.create_dataset('roi_color_map',
            data=serialize_extract_roi_list(color_map))

    print('wrote color map')

    with h5py.File(args.video_path, 'r') as in_file:
        img_data = in_file['data'][()]
    img_data = np.max(img_data, axis=0).astype(float)
    img_data = img_data - img_data.min()
    q01, q999 = np.quantile(img_data, (0.1, 0.999))
    img_data = np.where(img_data>q01, img_data, q01)
    img_data = np.where(img_data<q999, img_data, q999)

    img_data = np.round(255.0*(img_data/img_data.max()))
    img_data = img_data.astype(np.uint8)

    with h5py.File(args.out_path, 'a') as out_file:
        out_file.create_dataset('max_projection', data=img_data)
    del img_data
    print('wrote max projection image')

    scaled_video = read_and_scale(
                      video_path=pathlib.Path(args.video_path),
                      origin=(0,0),
                      frame_shape=(512, 512),
                      quantiles=(0.1, 0.999))

    print('scaled video')

    #upsized_video = upscale_video_frame(scaled_video, 4)
    #del scaled_video
    #print('upscaled video -- ',upsized_video.dtype)
    with h5py.File(args.out_path, 'a') as out_file:
        out_file.create_dataset('video_data', data=scaled_video)

    del scaled_video

    trace_lookup = get_traces(args.video_path, ophys_roi_list)
    with h5py.File(args.out_path, 'a') as out_file:
        group = out_file.create_group('traces')
        for roi_id in trace_lookup:
            group.create_dataset(str(roi_id), data=trace_lookup[roi_id])
