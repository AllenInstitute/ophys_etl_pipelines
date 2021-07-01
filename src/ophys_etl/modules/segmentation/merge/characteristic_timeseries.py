"""
This module contains the code that roi_merging.py uses to find
the single timeseries characterizing each ROI
"""
from typing import List, Tuple, Union, Set, Dict
import sys

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

import numpy as np
import multiprocessing
import multiprocessing.managers
from ophys_etl.modules.segmentation.merge.utils import (
    _winnow_process_list)

from ophys_etl.modules.segmentation.\
    merge.roi_time_correlation import (
        get_characteristic_timeseries,
        get_characteristic_timeseries_parallel)

import logging

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


class CharacteristicTimeseries(TypedDict):
    area: float  # tracked so we don't needlessly update after small mergers
    timeseries: np.ndarray


def _update_timeseries_lookup_per_pix(
        needed_rois:  Union[List[int], Set[int]],
        sub_video_lookup: Dict[int, np.ndarray],
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

    sub_video_lookup: Dict[int, np.ndarray]
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
    output: Dict[int, np.ndarray]
        Maps ROI ID to the characteristic timeseries for that ROI
    """
    if len(needed_rois) == 0:
        return dict()

    final_output = dict()
    for roi_id in needed_rois:
        sub_video = sub_video_lookup[roi_id]
        final_output[roi_id] = get_characteristic_timeseries_parallel(
                                      sub_video,
                                      filter_fraction=filter_fraction,
                                      n_processors=n_processors)

    return final_output


def _get_characteristic_timeseries(
        roi_id_list: List[int],
        sub_video_lookup: Dict[int, np.ndarray],
        filter_fraction: float,
        output_dict: Union[dict,
                           multiprocessing.managers.DictProxy]) -> None:
    """
    Method to calculate the characteristic time series of an ROI
    and store it in a multiprocessing DictProxy

    Parameters
    ----------
    roi_id_list: List[int]

    sub_video_lookup: Dict[int, np.ndarray]
        Dict mapping ROI ID to sub-videos which have been
        flattened in space so that their shapes are
        (ntime, npixels)

    filter_fraction: float
        The fraction of timesteps (chosen to be the brightest) to
        keep when doing the correlation.

    output_dict: Union[dict, multiprocessing.managers.DictProxy]

    Returns
    -------
    None
        Results are stored in output_dict
    """
    for roi_id in roi_id_list:
        pixel = get_characteristic_timeseries(
                                    sub_video_lookup[roi_id],
                                    filter_fraction=filter_fraction)
        output_dict[roi_id] = pixel
    return None


def _update_timeseries_lookup(needed_rois: Union[List[int], Set[int]],
                              sub_video_lookup: Dict[int, np.ndarray],
                              filter_fraction: float,
                              n_processors: int) -> Dict[int, np.ndarray]:
    """
    Method to calculate the characteristic time series of
    ROIs, using multiprocessing to process multiple ROIs
    in parallel

    Parameters
    ----------
    needed_rois: Union[List[int], Set[int]]
        List or set containing the ROI IDs of the ROIs whose
        characteristic timeseries are being calculated

    sub_video_lookup: Dict[int, np.ndarray]
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
    output: Dict[int, np.ndarray]
        Maps ROI ID to the characteristic timeseries for that ROI
    """
    if len(needed_rois) == 0:
        return dict()

    chunksize = len(needed_rois)//(n_processors-1)
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
        p = multiprocessing.Process(target=_get_characteristic_timeseries,
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


def update_timeseries_lookup(
        merger_candidates: List[Tuple[int, int]],
        timeseries_lookup: Dict[int, CharacteristicTimeseries],
        sub_video_lookup: Dict[int, np.ndarray],
        filter_fraction: float,
        n_processors: int,
        size_threshold: int = 500) -> Dict[int, CharacteristicTimeseries]:
    """
    Take a list of candidate merger ROI IDs and timeseries_lookup dict.
    Update timeseries_lookup dict with the key pixel time series for
    any pixels that are missing from the lookup table.

    Parameters
    ----------
    merger_candidates: List[Tuple[int, int]]
        A list of tuples representing potential ROI mergers

    timeseries_lookup: Dict[int, dict]
        A dict mapping ROI IDs to the characteristic timeseries
        associated with those ROIs (see Notes for more details)

    sub_video_lookup: Dict[int, np.ndarray]
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
    timeseries_lookup: dict
        Updated with any key pixels that need to be added.

    Notes
    -----
    timeseries_lookup actually maps ROI IDs to another dict

    timeseries_lookup[roi_id]['timeseries'] is the characteristic
    time series of the ROI

    timeseries_lookup[roi_id]['area'] is the area of the ROI
    at the time when the characteristic time series was calculated
    """

    # if the ROIs are larger than 500 pixels in area,
    # their characteristic time series will be calculated
    # by a method that is parallelized at the pixel, rather
    # than the ROI level
    needed_big_rois = set()
    needed_small_rois = set()

    roi_to_consider = {roi_id
                       for pair in merger_candidates
                       for roi_id in pair}

    for roi_id in roi_to_consider:
        if roi_id not in timeseries_lookup:
            area = sub_video_lookup[roi_id].shape[1]
            if area >= size_threshold:
                needed_big_rois.add(roi_id)
            else:
                needed_small_rois.add(roi_id)

    new_small_pixels = _update_timeseries_lookup(
                                     needed_small_rois,
                                     sub_video_lookup,
                                     filter_fraction,
                                     n_processors)

    new_big_pixels = _update_timeseries_lookup_per_pix(
                             needed_big_rois,
                             sub_video_lookup,
                             filter_fraction,
                             n_processors)

    for n in new_big_pixels:
        timeseries_lookup[n] = CharacteristicTimeseries(
                                       area=sub_video_lookup[n].shape[1],
                                       timeseries=new_big_pixels[n])
    for n in new_small_pixels:
        timeseries_lookup[n] = CharacteristicTimeseries(
                                     area=sub_video_lookup[n].shape[1],
                                     timeseries=new_small_pixels[n])

    return timeseries_lookup
