"""
This module contains the code that roi_merging.py uses to calculate the
merger metric for each candidate merger
"""
from typing import List, Tuple
import multiprocessing
import multiprocessing.managers
from ophys_etl.modules.segmentation.merge.utils import (
    _winnow_process_list)

from ophys_etl.modules.segmentation.merge.roi_time_correlation import (
        calculate_merger_metric)


def _calculate_merger_metric(
        input_pair_list: List[Tuple[int, int]],
        video_lookup: dict,
        timeseries_lookup: dict,
        self_corr_lookup: dict,
        filter_fraction: float,
        output_dict: multiprocessing.managers.DictProxy) -> None:
    """
    Calculate the merger metric for pairs of ROIs

    Parameters
    ----------
    input_pair_list: List[Tuple[int, int]]
        List of ROI ID pairs that are being considered for merger

    video_lookup: dict
        A dict that maps ROI ID to sub_videos (where sub_videos
        are flattened in space so that their shapes are
        (ntime, npixels))

    timeseries_lookup: dict
        A dict that maps ROI ID to the characteristic pixels
        associated with ROIs (i.e. their characteristic
        timeseries)

    self_corr_lookup: dict
        A dict that maps ROI ID to the (mu, std) tuples characterizing
        the Gaussian distributions of ROI pixels with their own
        characteristic time series

    filter_fraction: float
        The fraction of brightest timesteps to keep when correlating pixels

    output_dict: multiprocessing.managers.DictProxy
        The dict where results will be stored. Keys are the ROI ID
        pair tuples. Values are the merger metric for that pair.

    Returns
    -------
    None

    Notes
    -----
    timeseries_lookup actually maps ROI ID to another dict.

    timeseries_lookup[roi_id]['timeseries'] is the time series associated
    with the roi_id

    timeseries_lookup[roi_id]['timeseries']['area'] is the area of the
    ROI when the key pixel time series was calculated (tracked so that
    we don't spend too much time re-calculating these when the ROIs
    change a very little)
    """
    for input_pair in input_pair_list:

        video0 = video_lookup[input_pair[0]]
        video1 = video_lookup[input_pair[1]]

        area0 = video0.shape[1]
        area1 = video1.shape[1]

        if area0 < 2 or area0 < 0.5*area1:
            metric01 = -999.0
        else:
            metric01 = calculate_merger_metric(
                             self_corr_lookup[input_pair[0]],
                             timeseries_lookup[input_pair[0]]['timeseries'],
                             video1,
                             filter_fraction=filter_fraction)

        if area1 < 2 or area1 < 0.5*area0:
            metric10 = -999.0
        else:
            metric10 = calculate_merger_metric(
                             self_corr_lookup[input_pair[1]],
                             timeseries_lookup[input_pair[1]]['timeseries'],
                             video0,
                             filter_fraction=filter_fraction)

        metric = max(metric01, metric10)
        output_dict[input_pair] = metric
    return None


def get_merger_metric_from_pairs(
        potential_mergers: List[Tuple[int, int]],
        video_lookup: dict,
        timeseries_lookup: dict,
        self_corr_lookup: dict,
        filter_fraction: float,
        n_processors: int) -> dict:
    """
    Calculate the merger metric for pairs of ROIs

    Parameters
    ----------
    input_pair_list: List[Tuple[int, int]]
        List of ROI ID pairs that are being considered for merger

    video_lookup: dict
        A dict that maps ROI ID to sub_videos (where sub_videos
        are flattened in space so that their shapes are
        (ntime, npixels))

    timeseries_lookup: dict
        A dict that maps ROI ID to the characteristic timeseries
        associated with ROIs

    self_corr_lookup: dict
        A dict that maps ROI ID to the (mu, std) tuples characterizing
        the Gaussian distributions of ROI pixels with their own
        characteristic time series

    filter_fraction: float
        The fraction of brightest timesteps to keep when correlating pixels

    n_processors: int
        Number of processors to invoke with multiprocessing

    Returns
    -------
    output: dict
        Maps a tuple of ROI IDs to the merger metric for that
        potential merger.

    Notes
    -----
    timeseries_lookup actually maps ROI ID to another dict.

    timeseries_lookup[roi_id]['timeseries'] is the time series associated
    with the roi_id

    timeseries_lookup[roi_id]['timeseries']['area'] is the area of the
    ROI when the key pixel time series was calculated (tracked so that
    we don't spend too much time re-calculating these when the ROIs
    change a very little)
    """

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
        this_video = {}
        this_pixel = {}
        this_corr = {}
        for roi_id in this_roi_id:
            this_video[roi_id] = video_lookup[roi_id]
            this_pixel[roi_id] = timeseries_lookup[roi_id]
            this_corr[roi_id] = self_corr_lookup[roi_id]

        args = (chunk,
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
