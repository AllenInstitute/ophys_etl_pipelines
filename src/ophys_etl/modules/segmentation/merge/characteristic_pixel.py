"""
This module contains the code that the ROI merging module uses to find
the single timeseries characterizing each ROI
"""
from typing import List, Tuple, Union, Set
import multiprocessing
import multiprocessing.managers
from ophys_etl.modules.segmentation.merge.utils import (
    _winnow_process_list)

from ophys_etl.modules.segmentation.\
    merge.roi_time_correlation import (
        get_brightest_pixel,
        get_brightest_pixel_parallel)

import logging

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def _update_key_pixel_lookup_per_pix(
        needed_rois:  Union[List[int], Set[int]],
        sub_video_lookup: dict,
        filter_fraction: float,
        n_processors: int) -> dict:
    """
    Method to calculate the characteristic time series of
    ROIs, using multiprocessing to parallelize each ROI
    on the pixel level.

    Parameters
    ----------
    needed_rois: Union[List[int], Set[int]]
        List or set containing the ROI IDs of the ROIs whose
        characteristic timeseries are being calculated

    sub_video_lookup: dict
        Dict mapping ROI ID to sub-videos which have been
        flattened in space so that their shapes are
        (ntime, npixels)

    filter_fraction: float
        The fraction of timesteps (chosen to be the brightest) to
        keep when doing the correlation.

    n_processors: int
        Number of processors to invoke with multiprocessing

    Returns
    -------
    output: dict
        Maps ROI ID to the characteristic timeseries for that ROI
    """

    final_output = {}
    for roi_id in needed_rois:
        sub_video = sub_video_lookup[roi_id]
        final_output[roi_id] = get_brightest_pixel_parallel(
                                      sub_video,
                                      filter_fraction=filter_fraction,
                                      n_processors=n_processors)

    return final_output


def _get_brightest_pixel(
        roi_id_list: List[int],
        sub_video_lookup: dict,
        filter_fraction: float,
        output_dict: multiprocessing.managers.DictProxy) -> dict:
    """
    Method to calculate the characteristic time series of an ROI
    and store it in a multiprocessing DictProxy

    Parameters
    ----------
    roi_id_list: List[int]

    sub_video_lookup: dict
        Dict mapping ROI ID to sub-videos which have been
        flattened in space so that their shapes are
        (ntime, npixels)

    filter_fraction: float
        The fraction of timesteps (chosen to be the brightest) to
        keep when doing the correlation.

    output_dict: multiprocessing.managers.DictProxy

    Returns
    -------
    None
        Results are stored in output_dict
    """
    for roi_id in roi_id_list:
        pixel = get_brightest_pixel(sub_video_lookup[roi_id],
                                    filter_fraction=filter_fraction)
        output_dict[roi_id] = pixel
    return None


def _update_key_pixel_lookup(needed_rois: Union[List[int], Set[int]],
                             sub_video_lookup: dict,
                             filter_fraction: float,
                             n_processors: int) -> dict:
    """
    Method to calculate the characteristic time series of
    ROIs, using multiprocessing to process multiple ROIs
    in parallel

    Parameters
    ----------
    needed_rois: Union[List[int], Set[int]]
        List or set containing the ROI IDs of the ROIs whose
        characteristic timeseries are being calculated

    sub_video_lookup: dict
        Dict mapping ROI ID to sub-videos which have been
        flattened in space so that their shapes are
        (ntime, npixels)

    filter_fraction: float
        The fraction of timesteps (chosen to be the brightest) to
        keep when doing the correlation.

    n_processors: int
        Number of processors to invoke with multiprocessing

    Returns
    -------
    output: dict
        Maps ROI ID to the characteristic timeseries for that ROI
    """
    chunksize = len(needed_rois)//(4*n_processors-1)
    chunksize = max(chunksize, 1)
    mgr = multiprocessing.Manager()
    output_dict = mgr.dict()
    process_list = []
    needed_rois = list(needed_rois)
    for i0 in range(0, len(needed_rois), chunksize):
        chunk = needed_rois[i0:i0+chunksize]
        this_video = {}
        for roi_id in chunk:
            this_video[roi_id] = sub_video_lookup[roi_id]
        args = (chunk,
                this_video,
                filter_fraction,
                output_dict)
        p = multiprocessing.Process(target=_get_brightest_pixel,
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


def update_key_pixel_lookup(merger_candidates: List[Tuple[int, int]],
                            key_pixel_lookup: dict,
                            sub_video_lookup: dict,
                            filter_fraction: float,
                            n_processors: int,
                            size_threshold: int = 500) -> dict:
    """
    Take a list of candidate merger ROI IDs and key_pixel_lookup dict.
    Update key_pixel_lookup dict with the key pixel time series for
    any pixels that are missing from the lookup table.

    Parameters
    ----------
    merger_candidates: List[Tuple[int, int]]
        A list of tuples representing potential ROI mergers

    key_pixel_lookup: dict
        A dict mapping ROI IDs to the characteristic timeseries
        (the "key pixels") associated with those ROIs

    sub_video_lookup: dict
        A dict mapping ROI IDs to sub-videos which have been
        flattened in space so that their shapes are (ntime, npixels)

    filter_fraction: float
        The fraction of timesteps (chosen to be the brightest) to
        keep when doing the correlation.

    n_processors: int
        The number of processors to invoke with multiprocessing

    size_threshold: int
        The area at which an ROI gets parallelized at the pixel level
        (default=500)

    Returns
    -------
    key_pixel_lookup: dict
        Updated with any key pixels that need to be added.

    Notes
    -----
    key_pixel_lookup actually maps ROI IDs to another dict

    key_pixel_lookup[roi_id]['key_pixel'] is the characteristic
    time series of the ROI

    key_pixel_lookup[roi_id]['area'] is the area of the ROI
    at the time when the characteristic time series was calculated
    """

    # if the ROIs are larger than 500 pixels in area,
    # their characteristic time series will be calculated
    # by a method that is parallelized at the pixel, rather
    # than the ROI level
    needed_big_rois = set()
    needed_small_rois = set()

    roi_to_consider = set()
    for pair in merger_candidates:
        roi_to_consider.add(pair[0])
        roi_to_consider.add(pair[1])

    for roi_id in roi_to_consider:
        if roi_id not in key_pixel_lookup:
            area = sub_video_lookup[roi_id].shape[1]
            if area >= size_threshold:
                needed_big_rois.add(roi_id)
            else:
                needed_small_rois.add(roi_id)

    new_small_pixels = {}
    if len(needed_small_rois) > 0:
        new_small_pixels = _update_key_pixel_lookup(
                                             needed_small_rois,
                                             sub_video_lookup,
                                             filter_fraction,
                                             n_processors)
    new_big_pixels = {}
    if len(needed_big_rois) > 0:
        logger.info('CALLING PIXEL-PARALLELIZED CORRELATION on '
                    f'{len(needed_big_rois)} ROIs')
        new_big_pixels = _update_key_pixel_lookup_per_pix(
                             needed_big_rois,
                             sub_video_lookup,
                             filter_fraction,
                             n_processors)

    for n in new_big_pixels:
        key_pixel_lookup[n] = {'area': sub_video_lookup[n].shape[1],
                               'key_pixel': new_big_pixels[n]}
    for n in new_small_pixels:
        key_pixel_lookup[n] = {'area': sub_video_lookup[n].shape[1],
                               'key_pixel': new_small_pixels[n]}

    return key_pixel_lookup
