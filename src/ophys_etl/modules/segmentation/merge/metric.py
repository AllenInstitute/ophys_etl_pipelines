from typing import List, Optional, Dict, Tuple, Union
from functools import partial
from itertools import combinations
import multiprocessing
import multiprocessing.managers
import numpy as np
from ophys_etl.modules.segmentation.merge.utils import (
    _winnow_process_list)
from ophys_etl.modules.segmentation.merge.roi_types import (
    SegmentationROI)

from ophys_etl.modules.segmentation.merge.roi_time_correlation import (
        calculate_merger_metric)


def _calculate_merger_metric(
        input_pair_list: List[Tuple[int, int]],
        roi_lookup: dict,
        video_lookup: dict,
        pixel_lookup: dict,
        self_corr_lookup: dict,
        filter_fraction: float,
        output_dict: multiprocessing.managers.DictProxy) -> None:
    """
    Calculate the merger metric for a pair of ROIs

    Parameters
    ----------
    input_pair: Tuple[int, int]
        pair of roi_ids to consider for merging

    roi_lookup: dict
        Maps roi_id to SegmentationROI

    video_lookup: dict
        Maps roi_id to sub_video

    filter_fraction: float
        The fraction of brightest timesteps to keep when correlating pixels

    Returns
    -------
    result: Tuple[int, int, float]
        The roi_ids of the pair and the largest value of the
        merger metric yielded by calling calculate_merger_metric on
        both permutations of the ROIs [(roi0, roi1) and (roi1, roi0)]
    """
    for input_pair in input_pair_list:

        roi0 = roi_lookup[input_pair[0]]
        roi1 = roi_lookup[input_pair[1]]

        if roi0.area < 2 or roi0.area < 0.5*roi1.area:
            metric01 = -999.0
        else:
            metric01 = calculate_merger_metric(
                             self_corr_lookup[input_pair[0]],
                             pixel_lookup[input_pair[0]]['key_pixel'],
                             video_lookup[input_pair[1]],
                             filter_fraction=filter_fraction)

        if roi1.area < 2 or roi1.area < 0.5*roi0.area:
            metric10 = -999.0
        else:
            metric10 = calculate_merger_metric(
                             self_corr_lookup[input_pair[1]],
                             pixel_lookup[input_pair[1]]['key_pixel'],
                             video_lookup[input_pair[0]],
                             filter_fraction=filter_fraction)

        metric = max(metric01, metric10)
        output_dict[input_pair] = metric
    return None


def get_merger_metric(potential_mergers,
                      roi_lookup,
                      video_lookup,
                      pixel_lookup,
                      self_corr_lookup,
                      filter_fraction,
                      n_processors):

    mgr = multiprocessing.Manager()
    output_dict = mgr.dict()
    n_pairs = len(potential_mergers)
    chunksize = n_pairs//(4*n_processors-1)
    chunksize = max(chunksize, 1)
    process_list = []
    for i0 in range(0, n_pairs, chunksize):
        chunk = potential_mergers[i0:i0+chunksize]
        this_roi_id = set()
        for pair in chunk:
            this_roi_id.add(pair[0])
            this_roi_id.add(pair[1])
        this_roi = {}
        this_video = {}
        this_pixel = {}
        this_corr = {}
        for roi_id in this_roi_id:
            this_roi[roi_id] = roi_lookup[roi_id]
            this_video[roi_id] = video_lookup[roi_id]
            this_pixel[roi_id] = pixel_lookup[roi_id]
            this_corr[roi_id] = self_corr_lookup[roi_id]

        args = (chunk,
                this_roi,
                this_video,
                this_pixel,
                this_corr,
                filter_fraction,
                output_dict)

        p = multiprocessing.Process(target=_calculate_merger_metric,
                                    args=args)
        p.start()
        process_list.append(p)
        while len(process_list) > 0 and len(process_list) >= (n_processors-1):
            process_list = _winnow_process_list(process_list)
    for p in process_list:
        p.join()

    final_output = {}
    k_list = list(output_dict.keys())
    for k in k_list:
        final_output[k] = output_dict.pop(k)

    return final_output
