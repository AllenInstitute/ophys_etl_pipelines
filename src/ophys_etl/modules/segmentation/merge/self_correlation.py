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
    get_self_correlation)


def _self_correlate_chunk(roi_id_list,
                          sub_video_lookup,
                          key_pixel_lookup,
                          filter_fraction,
                          output_dict):
    for roi_id in roi_id_list:
        result = get_self_correlation(sub_video_lookup[roi_id],
                                      key_pixel_lookup[roi_id]['key_pixel'],
                                      filter_fraction)
        output_dict[roi_id] = result
    return None


def update_self_correlation(merger_candidates,
                            sub_video_lookup,
                            key_pixel_lookup,
                            filter_fraction,
                            self_corr_lookup,
                            n_processors):
    roi_id_list = set()
    for pair in merger_candidates:
        roi_id_list.add(pair[0])
        roi_id_list.add(pair[1])
    roi_id_list = list(roi_id_list)

    mgr = multiprocessing.Manager()
    output_dict = mgr.dict()
    process_list = []
    chunksize = max(1, len(roi_id_list)//(n_processors-1))
    for i0 in range(0, len(roi_id_list), chunksize):
        chunk = roi_id_list[i0:i0+chunksize]
        this_video = {}
        this_pixel = {}
        for roi_id in chunk:
            this_video[roi_id] = sub_video_lookup[roi_id]
            this_pixel[roi_id] = key_pixel_lookup[roi_id]
        args = (chunk,
                this_video,
                this_pixel,
                filter_fraction,
                output_dict)
        p = multiprocessing.Process(target=_self_correlate_chunk,
                                    args=args)
        p.start()
        process_list.append(p)
        while len(process_list)>0 and len(process_list)>=(n_processors-1):
            process_list = _winnow_process_list(process_list)
    for p in process_list:
        p.join()

    k_list = list(output_dict.keys())
    for k in k_list:
        self_corr_lookup[k] = output_dict.pop(k)
    return self_corr_lookup
