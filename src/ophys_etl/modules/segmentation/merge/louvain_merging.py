from typing import Tuple, Dict, Union, List
import pathlib
import numpy as np

import logging
import time

from ophys_etl.utils.array_utils import (
    pairwise_distances)

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.utils.roi_utils import (
    pixel_list_to_extract_roi,
    extract_roi_to_ophys_roi)

from ophys_etl.modules.segmentation.merge.louvain_utils import (
    correlate_all_pixels,
    _do_louvain_clustering)


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def do_louvain_clustering_on_rois(
        roi_list: List[OphysROI],
        full_video: np.ndarray,
        kernel_size: Union[float, None],
        filter_fraction: float,
        n_processors: int,
        scratch_dir: pathlib.Path) -> Tuple[List[OphysROI],
                                            List[List[int]]]:
    """
    full_video is in (n_time, n_rows, n_cols)
    output array is of form [[dst, src], [dst, src], [dst, src]]
    """
    n_time = full_video.shape[0]
    all_pixel_set = set()
    for roi in roi_list:
        all_pixel_set = all_pixel_set.union(roi.global_pixel_set)
    n_pixels = len(all_pixel_set)
    sub_video = np.zeros((n_time, n_pixels), dtype=float)
    index_to_pixel = []
    roi_id_array = []

    # don't really know how to handle overlaps here
    already_added_pixels = set()
    i_pixel = 0
    for roi in roi_list:
        for pixel in roi.global_pixel_array:
            pixel = tuple(pixel)
            if pixel in already_added_pixels:
                continue
            trace = full_video[:, pixel[0], pixel[1]]
            sub_video[:, i_pixel] = trace
            index_to_pixel.append(pixel)
            roi_id_array.append(roi.roi_id)
            i_pixel += 1
            already_added_pixels.add(pixel)

    roi_id_array = np.array(roi_id_array)

    if kernel_size is None:
        pixel_distances = None
    else:
        pixel_coord_array = np.array([[p[0], p[1]]
                                      for p in index_to_pixel])
        pixel_distances = pairwise_distances(pixel_coord_array)

    t0 = time.time()
    correlation_matrix = correlate_all_pixels(
                            sub_video,
                            filter_fraction,
                            n_processors,
                            scratch_dir,
                            pixel_distances=pixel_distances,
                            kernel_size=kernel_size)

    duration = time.time()-t0
    logger.info(f'calculated correlation matrix in {duration:.2f} seconds')

    (merged_roi_id_array,
     merger_history) = _do_louvain_clustering(roi_id_array,
                                              correlation_matrix)

    new_roi_list = []
    for roi_id in np.unique(merged_roi_id_array):
        pixel_list = [index_to_pixel[ii]
                      for ii in np.where(merged_roi_id_array == roi_id)[0]]
        new_extract_roi = pixel_list_to_extract_roi(pixel_list, roi_id)
        new_ophys_roi = extract_roi_to_ophys_roi(new_extract_roi)
        new_roi_list.append(new_ophys_roi)

    output_merger_history = [[merger_history[roi_id], roi_id]
                             for roi_id in merger_history]

    return (new_roi_list,
            output_merger_history)
