import matplotlib.figure as mplt_figure

import argparse
import pathlib
import h5py
import json
import numpy as np
import time

from ophys_etl.modules.segmentation.processing_log import (
    SegmentationProcessingLog)
from ophys_etl.modules.segmentation.merge.louvain_utils import (
    find_roi_clusters)
from ophys_etl.modules.segmentation.merge.louvain_merging import (
    do_louvain_clustering_on_rois)
from ophys_etl.modules.segmentation.qc_utils.roi_comparison_utils import(
    get_roi_color_map)
from ophys_etl.modules.segmentation.graph_utils.conversion import(
    graph_to_img)
from ophys_etl.modules.segmentation.utils.roi_utils import(
    extract_roi_to_ophys_roi,
    ophys_roi_to_extract_roi)
import os

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
    add_roi_boundary_to_img)
from ophys_etl.modules.segmentation.qc_utils.video_utils import (
    scale_video_to_uint8)


def add_rois(background_img, roi_list, roi_color_map):
    img = np.copy(background_img)
    for roi in roi_list:
        assert roi.roi_id in roi_color_map
        if not roi.valid_roi:
            continue
        img = add_roi_boundary_to_img(
                  img,
                  roi,
                  roi_color_map[roi.roi_id], 1.)
    return img

def generate_roi_figure(axis,
                        roi_list,
                        plot_title,
                        roi_color_map,
                        background_img,
                        row_bounds=None,
                        col_bounds=None):

    img = add_rois(background_img, roi_list, roi_color_map)

    if row_bounds is not None:
        img = img[row_bounds[0]:row_bounds[1],:]
    if col_bounds is not None:
        img = img[:, col_bounds[0]:col_bounds[1]]

    axis.imshow(img)
    axis.set_title(plot_title, fontsize=20)
    return axis


if __name__ == "__main__":
    global_t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--video_path', type=str)
    parser.add_argument('--graph_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--plot_path', type=str)
    parser.add_argument('--scratch_dir', type=str)
    parser.add_argument('--n_processors', type=int, default=8)
    parser.add_argument('--kernel_size', type=float, default=None)

    args = parser.parse_args()

    splog = SegmentationProcessingLog(args.log_path, read_only=True)
    assert os.path.isfile(args.video_path)
    assert os.path.isfile(args.graph_path)
    assert os.path.isdir(args.scratch_dir)

    raw_roi_list = splog.get_rois_from_group(group_name='detect')
    input_roi_list = [extract_roi_to_ophys_roi(roi) for roi in
                      raw_roi_list]

    t0 = time.time()
    roi_clusters = find_roi_clusters(input_roi_list)
    duration = time.time()-t0
    print(f'finding clusters took {duration:.2f} seconds')

    with h5py.File(args.video_path,'r') as in_file:
        full_video = in_file['data'][()]
    print('read in full_video')

    merged_roi_list = []
    n_clusters = len(roi_clusters)
    for i_cluster, cluster in enumerate(roi_clusters):
        if len(cluster) == 1:
            merged_roi_list.append(cluster[0])
            continue
        area = 0
        for roi in cluster:
            area += roi.area
        t0 = time.time()
        print(f'starting cluster {i_cluster} of {n_clusters} '
              f'with area {area:.2e} -- {len(cluster)} ROIs')
        (new_cluster,
         history) = do_louvain_clustering_on_rois(
                         cluster,
                         full_video,
                         args.kernel_size,
                         0.2,
                         args.n_processors,
                         pathlib.Path(args.scratch_dir))
        duration = time.time()-t0
        print(f'done in {duration:.2e} seconds -- {len(new_cluster)} ROIs')
        for roi in new_cluster:
            merged_roi_list.append(roi)

    with open(args.output_path, 'w') as out_file:
        e_list = [ophys_roi_to_extract_roi(roi) for roi in
                  merged_roi_list]
        out_file.write(json.dumps(e_list, indent=2))
    print('serialized ROIs')

    original_map = get_roi_color_map(input_roi_list)
    merged_map = get_roi_color_map(merged_roi_list)

    raw_img = graph_to_img(pathlib.Path(args.graph_path))
    minval, maxval = np.quantile(raw_img, (0.1, 0.999))

    raw_background_img = scale_video_to_uint8(
                           raw_img, minval, maxval)

    background_img = np.zeros((raw_background_img.shape[0],
                                   raw_background_img.shape[1],
                                   3), dtype=np.uint8)
    for ic in range(3):
        background_img[:, :, ic] = raw_background_img

    blank = np.zeros(background_img.shape, dtype=np.uint8)

    fig = mplt_figure.Figure(figsize=(20,20))
    axes = [fig.add_subplot(2,2,ii) for ii in range(1,5)]

    axes[0] = generate_roi_figure(
                 axes[0],
                 input_roi_list,
                 'input ROIs',
                 original_map,
                 background_img)    
    axes[2] = generate_roi_figure(
                 axes[2],
                 input_roi_list,
                 'input ROIs',
                 original_map,
                 blank)    

    axes[1] = generate_roi_figure(
                 axes[1],
                 merged_roi_list,
                 'merged ROIs',
                 merged_map,
                 background_img)    
    axes[3] = generate_roi_figure(
                 axes[3],
                 merged_roi_list,
                 'merged ROIs',
                 merged_map,
                 blank)    


    for a in axes:
        for ii in range(0, blank.shape[0], 64):
            a.axhline(ii, color='w', alpha=0.5)
            a.axvline(ii, color='w', alpha=0.5)
    fig.tight_layout()
    fig.savefig(args.plot_path)

    duration = time.time()-global_t0
    print(f'whole thing took {duration:.2f} seconds')
