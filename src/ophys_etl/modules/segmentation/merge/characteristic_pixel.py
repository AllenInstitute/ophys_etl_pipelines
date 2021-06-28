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

from ophys_etl.modules.segmentation.\
    merge.roi_time_correlation import (
        get_brightest_pixel,
        get_brightest_pixel_parallel)


def _get_brightest_pixel(roi_id_list: List[int],
                         sub_video_lookup: dict,
                         output_dict: multiprocessing.managers.DictProxy):
    for roi_id in roi_id_list:
        pixel = get_brightest_pixel(sub_video_lookup[roi_id])

        output_dict[roi_id] = pixel


def _update_key_pixel_lookup_per_pix(
        needed_pixels,
        sub_video_lookup,
        n_processors):

    final_output = {}
    for ipix in needed_pixels:
        sub_video = sub_video_lookup[ipix]
        final_output[ipix] = get_brightest_pixel_parallel(
                                      sub_video,
                                      n_processors=n_processors)

    return final_output


def _update_key_pixel_lookup(needed_pixels,
                             roi_lookup,
                             sub_video_lookup,
                             n_processors):
    chunksize = len(needed_pixels)//(4*n_processors-1)
    chunksize = max(chunksize, 1)
    mgr = multiprocessing.Manager()
    output_dict = mgr.dict()
    process_list = []
    needed_pixels = list(needed_pixels)
    for i0 in range(0, len(needed_pixels), chunksize):
        chunk = needed_pixels[i0:i0+chunksize]
        this_video = {}
        for roi_id in chunk:
            this_video[roi_id] = sub_video_lookup[roi_id]
        args = (chunk,
                this_video,
                output_dict)
        p = multiprocessing.Process(target=_get_brightest_pixel,
                                    args=args)
        p.start()
        process_list.append(p)
        while len(process_list)>0 and len(process_list)>=(n_processors-1):
            process_list = _winnow_process_list(process_list)
    for p in process_list:
        p.join()
    final_output = {}
    k_list = list(output_dict.keys())
    for k in k_list:
        final_output[k] = output_dict.pop(k)
    return final_output


def update_key_pixel_lookup(merger_candidates,
                            roi_lookup,
                            pixel_lookup,
                            sub_video_lookup,
                            n_processors):

    needed_big_pixels = set()
    needed_small_pixels = set()

    roi_to_consider = set()
    for pair in merger_candidates:
        roi_to_consider.add(pair[0])
        roi_to_consider.add(pair[1])

    for roi_id in roi_to_consider:
        needs_update = False
        if roi_id not in pixel_lookup:
            needs_update = True
            s = roi_lookup[roi_id].area
            if s >= 500:
                needed_big_pixels.add(roi_id)
            else:
                needed_small_pixels.add(roi_id)

    new_small_pixels = {}
    if len(needed_small_pixels) > 0:
        new_small_pixels = _update_key_pixel_lookup(
                                             needed_small_pixels,
                                             roi_lookup,
                                             sub_video_lookup,
                                             n_processors)
    new_big_pixels = {}
    if len(needed_big_pixels) > 0:
        logger.info(f'CALLING BIG PIXEL CORRELATION on {len(needed_big_pixels)} ROIs')
        new_big_pixels = _update_key_pixel_lookup_per_pix(
                             needed_big_pixels,
                             sub_video_lookup,
                             n_processors)

    for n in new_big_pixels:
        pixel_lookup[n] = {'area': roi_lookup[n].area,
                           'key_pixel': new_big_pixels[n]}
    for n in new_small_pixels:
        pixel_lookup[n] = {'area': roi_lookup[n].area,
                           'key_pixel': new_small_pixels[n]}

    return pixel_lookup
