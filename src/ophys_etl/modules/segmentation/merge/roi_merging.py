from typing import List, Dict, Tuple, Set
import numpy as np
import copy

from ophys_etl.modules.segmentation.\
    merge.roi_utils import (
        merge_rois,
        sub_video_from_roi)

from ophys_etl.modules.segmentation.merge.candidates import (
    create_neighbor_lookup)

from ophys_etl.modules.segmentation.merge.metric import (
    get_merger_metric_from_pairs)

from ophys_etl.modules.segmentation.merge.characteristic_timeseries import (
    update_timeseries_lookup,
    CharacteristicTimeseries)

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

import logging
import time


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def get_new_merger_candidates(
     neighbor_lookup: Dict[int, Set[int]],
     merger_to_metric: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
    """
    Find all of the candidate mergers for which we do not already have
    a merger metric calculated

    Parameters
    ----------
    neighbor_lookup: Dict[int, Set[int]]
        Dict mapping ROI ID to the list of ROI IDs of neighbors

    merger_to_metric: Dict[Tuple[int, int], float]
        Dict mapping a pair of ROI IDs to the merger metric value
        associated with that merger. This is meant to track the mergers
        we already have metric values for (i.e. that we do not need to
        recalculate)

    Returns
    -------
    merger_candidate: List[Tuple[int, int]]
        List of pairs of ROI IDs representing mergers we need to calculate
        the metric for
    """
    # find all pairs of ROIs that abut
    raw_merger_candidates = set()
    for roi_id_0 in neighbor_lookup:
        for roi_id_1 in neighbor_lookup[roi_id_0]:
            if roi_id_0 > roi_id_1:
                pair = (roi_id_0, roi_id_1)
            else:
                pair = (roi_id_1, roi_id_0)
            raw_merger_candidates.add(pair)

    # only need to calculate metrics for those that have
    # changed since the last iteration, though
    raw_merger_candidates = list(raw_merger_candidates)
    merger_candidates = []
    for pair in raw_merger_candidates:
        if pair not in merger_to_metric:
            merger_candidates.append(pair)
    return merger_candidates


def break_out_anomalous_rois(
        roi_lookup: Dict[int, OphysROI],
        anomalous_rois: Dict[int, OphysROI],
        anomalous_size: int) -> Tuple[Dict[int, OphysROI],
                                      Dict[int, OphysROI]]:
    """
    Move any ROIs whose areas exceed anomalous_size from
    roi_lookup to anomalous_rois

    Parameters
    ----------
    roi_lookup: Dict[int, OphysROI]

    anomalous_rois: Dict[int, OphysROI]

    anomalous_size: int

    Returns
    -------
    roi_lookup: Dict[int, OphysROI]
        Updated

    anomalous_rois: Dict[int, OphysROI]
        Updated
    """
    # break out any ROIs that are too large
    k_list = list(roi_lookup.keys())
    for roi_id in k_list:
        if roi_lookup[roi_id].area >= anomalous_size:
            roi = roi_lookup.pop(roi_id)
            roi.valid_roi = False
            anomalous_rois[roi_id] = roi

    return roi_lookup, anomalous_rois


def update_lookup_tables(
     roi_lookup: Dict[int, OphysROI],
     recently_merged: Set[int],
     neighbor_lookup: Dict[int, Set[int]],
     sub_video_lookup: Dict[int, np.ndarray],
     timeseries_lookup: Dict[int, CharacteristicTimeseries],
     merger_to_metric: Dict[Tuple[int, int], float]) -> Tuple[
                                          Dict[int, Set[int]],
                                          Dict[int, np.ndarray],
                                          Dict[int, CharacteristicTimeseries],
                                          Dict[Tuple[int, int], float]]:
    """
    Update lookup tables to reflect mergers, removing entries
    that are not longer valid or need to be recalculated

    Parameters
    ----------
    roi_lookup: Dict[int, OphysROI]

    recently_merged: Set[int]
        ROI IDs of ROIs that recently participated in mergers

    neighbor_lookup: Dict[int, Set[int]]
        Maps ROI ID to list of neighboring ROI IDs

    sub_video_lookup: Dict[int, np.ndarray]

    timeseries_lookup: Dict[int, CharacteristicTimeseries]

    merger_to_metric: Dict[Tuple[int, int], float]
        Maps a potential merger pair to the merger's associated
        metric value

    Returns
    -------
    neighbor_lookup: Dict[int, Set[int]]

    sub_video_lookup: Dict[int, np.ndarray]

    timeseries_lookup: Dict[int, CharacteristicTimeseries]

    merger_to_metric: Dict[Tuple[int, int], float]

    Notes
    -----
    ROIs that do not appear in roi_lookup are removed
    from lookup tables. ROIs whose area has changed by
    more than 5% since the timeseries was calculated
    are removed from timeseries_lookup (prompting them
    to be recalculated on the next iteration). ROIs that
    have been recently merged are removed from
    sub_video_lookup and merger_to_metric.
    """

    # work on copies; editing in place seems dangerous,
    # and this will likely not be the slowest part of the
    # code
    neighbor_lookup = copy.deepcopy(neighbor_lookup)
    sub_video_lookup = copy.deepcopy(sub_video_lookup)
    timeseries_lookup = copy.deepcopy(timeseries_lookup)
    merger_to_metric = copy.deepcopy(merger_to_metric)

    valid_roi_id = set(roi_lookup.keys())
    for table in (timeseries_lookup,
                  neighbor_lookup,
                  sub_video_lookup,
                  neighbor_lookup):
        k_list = list(table.keys())
        for k in k_list:
            if k not in valid_roi_id:
                table.pop(k)

    # remove an obsolete ROI IDs from neighbor lookup
    for roi_id in neighbor_lookup:
        new_set = neighbor_lookup[roi_id].intersection(valid_roi_id)
        neighbor_lookup[roi_id] = new_set

    # remove ROIs that have changed from sub_video and
    # merger metric lookup tables
    for roi_id in recently_merged:
        if roi_id in sub_video_lookup:
            sub_video_lookup.pop(roi_id)

    # remove ROIs whose areas
    # have significantly changed from timeseries_lookup
    # (calculating timeseries_lookup is expensive, so
    # we allow ROIs to grow a little before recalculating)
    k_list = list(timeseries_lookup.keys())
    for roi_id in k_list:
        area0 = timeseries_lookup[roi_id]['area']
        if roi_lookup[roi_id].area > 1.05*area0:
            timeseries_lookup.pop(roi_id)

    # remove all recently merged ROIs from the merger metric
    # lookup table (also all invalid ROIs)
    merger_keys = list(merger_to_metric.keys())
    for pair in merger_keys:
        if pair[0] in recently_merged or pair[1] in recently_merged:
            merger_to_metric.pop(pair)
        elif pair[0] not in valid_roi_id or pair[1] not in valid_roi_id:
            merger_to_metric.pop(pair)

    return (neighbor_lookup,
            sub_video_lookup,
            timeseries_lookup,
            merger_to_metric)


def _do_mergers(
        merger_to_metric: Dict[Tuple[int, int], float],
        roi_lookup: Dict[int, OphysROI],
        neighbor_lookup: Dict[int, Set[int]],
        have_been_absorbed: Set[int]) -> Tuple[Set[int],
                                               Dict[int, OphysROI],
                                               Dict[int, Set[int]],
                                               Set[int]]:
    """
    Actually perform ROI mergers

    Parameters
    ----------
    merger_to_metric: Dict[Tuple[int, int], float]
        Maps pair of ROI IDs to the merger metric value.
        **Should only contain mergers with a metric value
        that passes acceptance criterion***

    roi_lookup: Dict[int, OphysROI]

    neighbor_lookup: Dict[int, Set[int]]
        Maps ROI ID to the list of neighboring ROI IDs

    have_been_absorbed: Set[int]
        Set for keeping track of ROIs that were absorbed into
        other ROIs

    Returns
    -------
    recently_merged: Set[int]
        ROI IDs of ROIs that participated in mergers

    roi_lookup: Dict[int, OphysROI]
        Updated to reflect mergers

    neighbor_lookup: Dict[int, Set[int]]
       Updated to reflect mergers

    have_been_absorbed: Set[int]
        Updated to reflect mergers
    """
    valid_roi_id = set(roi_lookup.keys())
    potential_mergers = []
    merger_metrics = []
    for pair in merger_to_metric:
        potential_mergers.append(pair)
        merger_metrics.append(merger_to_metric[pair])

    # order mergers by metric in descending order
    potential_mergers = np.array(potential_mergers)
    merger_metrics = np.array(merger_metrics)
    sorted_indices = np.argsort(-1*merger_metrics)
    potential_mergers = potential_mergers[sorted_indices]

    recently_merged = set()
    wait_for_it = set()

    area_lookup = dict()
    for roi_id in roi_lookup:
        area_lookup[roi_id] = roi_lookup[roi_id].area

    for merger in potential_mergers:
        roi_id_0 = merger[0]
        roi_id_1 = merger[1]
        go_ahead = True

        # if one of the ROIs no longer exists because
        # it was absorbed into another, skip
        if roi_id_0 not in valid_roi_id:
            go_ahead = False
        if roi_id_1 not in valid_roi_id:
            go_ahead = False

        # if one of the ROIs has grown by 10% or more since
        # the start of this iteration, skip
        if go_ahead:
            if roi_id_0 in recently_merged:
                if roi_lookup[roi_id_0].area > 1.1*area_lookup[roi_id_0]:
                    go_ahead = False
            if roi_id_1 in recently_merged:
                if roi_lookup[roi_id_1].area > 1.1*area_lookup[roi_id_1]:
                    go_ahead = False

        # if one of the ROIs was in a potential pair that was skipped
        # for one of the reasons above, skip (in case that merger
        # really was the best merger for it; these mergers will
        # be reconsidered in the next iteration)
        if go_ahead:
            if roi_id_0 in wait_for_it:
                go_ahead = False
            if roi_id_1 in wait_for_it:
                go_ahead = False

        # mark these ROIs has having deferred a merger and continue
        # (if appropriate)
        if not go_ahead:
            wait_for_it.add(roi_id_0)
            wait_for_it.add(roi_id_1)
            continue

        # order by size
        if roi_lookup[roi_id_0].area > roi_lookup[roi_id_1].area:
            seed_id = roi_id_0
            child_id = roi_id_1
        else:
            seed_id = roi_id_1
            child_id = roi_id_0

        seed_roi = roi_lookup[seed_id]
        child_roi = roi_lookup[child_id]

        new_roi = merge_rois(seed_roi,
                             child_roi,
                             seed_roi.roi_id)

        # remove the ROI that was absorbed
        roi_lookup.pop(child_roi.roi_id)
        valid_roi_id.remove(child_roi.roi_id)

        # mark these ROIs as ROIs that have been merged
        have_been_absorbed.add(child_roi.roi_id)
        recently_merged.add(child_roi.roi_id)
        recently_merged.add(new_roi.roi_id)
        roi_lookup[seed_roi.roi_id] = new_roi

        # all ROIs that neighbored child_roi now neighbor new_roi
        severed_neighbors = neighbor_lookup.pop(child_roi.roi_id)
        severed_neighbors = severed_neighbors.intersection(valid_roi_id)
        for roi_id in severed_neighbors:
            if roi_id == new_roi.roi_id:
                continue
            neighbor_lookup[new_roi.roi_id].add(roi_id)
            neighbor_lookup[roi_id].add(new_roi.roi_id)

    return (recently_merged,
            roi_lookup,
            neighbor_lookup,
            have_been_absorbed)


def do_roi_merger(
      raw_roi_list: List[OphysROI],
      video_data: np.ndarray,
      n_processors: int,
      corr_acceptance: float,
      filter_fraction: float = 0.2) -> List[OphysROI]:
    """
    Merge ROIs based on a static image.

    Parameters
    ----------
    raw_roi_list: List[OphysROI]

    video_data: np.ndarray
        (ntime, nrows, ncols)

    n_processors: int
        The number of processors to invoke with multiprocessing

    corr_acceptance: float
        The pixel time correlation threshold for accepting a
        merger (see "Notes")

    filter_fraction: float
        Fraction of timesteps to keep when correlating pixels
        in sub_videos (default=0.2)

    Returns
    -------
    List[OphysROI]
        List of ROIs after merger. ROIs will have been cast
        to OphysROIs, but they have the same spatial
        information and API as OphysROIs

    Notes
    -----
    This algorithm works as follows:
    1) Find all pairs of ROIs that are neighbors (in this case, being
    a neighbor means physically abutting one another)

    2) Loop over all pairs of ROIs. For each pair, assess the validity
    of the merger by

        2a) Correlate all of the pixels in roi0 against a characteristic
        timeseries that is a weighted average of the pixels in roi0
        using the brightest filter_fraction of time steps in the
        characteristic timeseries. Use these correlations to construct
        a Gaussian distribution.

        2b) Correlate all of the pixels in roi1 against the characteristic
        timeseries for roi0 using the same timesteps.

        2c) Calculate the z-score of the correlations from (2b) using
        the Gaussian distribution from (2a).

        2d) Reverse roi0 and roi1 and repeat steps (2a-c). Evaluate the
        merger based on the highest resulting median z-score. If that
        z-score is greater than or equal to -1*corr_acceptance,
        the merger is valid.

    3) Rank the potential valid mergers based on the median z-sccore
    from step (2d). Move down the list of ranked mergers, merging ROIs.
    Once an ROI participates in the merger, it is removed from
    consideration until the next iteration.

    4) Repeat steps (1-3) until no more mergers occur.
    """

    anomalous_size = 800

    # create a lookup table of OphysROIs
    roi_lookup = dict()
    for roi in raw_roi_list:
        roi_lookup[roi.roi_id] = roi

    t0 = time.time()
    logger.info('starting merger')

    incoming_rois = list(roi_lookup.keys())

    have_been_absorbed = set()  # keep track of ROIs that were merged

    neighbor_lookup = create_neighbor_lookup(
                          roi_lookup,
                          n_processors)

    logger.info('found global merger candidates in '
                f'{time.time()-t0:.2f} seconds')

    # global lookup tables for per-ROI sub-videos and mappings
    # between potential merger pairs and merger metrics
    sub_video_lookup = dict()
    merger_to_metric = dict()
    timeseries_lookup = dict()

    # ROIs that are too large to be considered valid
    anomalous_rois = dict()

    logger.info(f'initially {len(roi_lookup)} ROIs')

    # pseudo-randomly order ROIs when calculating
    # expensive statistics to prevent the most expensive
    # ROIs from all landing on the same process
    shuffler = np.random.RandomState(11723412)

    keep_going = True
    while keep_going:

        # statistics on this pass for INFO messages
        n0 = len(roi_lookup)

        # fill in sub-video lookup for ROIs that changed
        # during the last iteration
        for roi_id in roi_lookup:
            if roi_id in sub_video_lookup:
                continue
            roi = roi_lookup[roi_id]
            sub_video_lookup[roi_id] = sub_video_from_roi(roi,
                                                          video_data)

        new_merger_candidates = get_new_merger_candidates(
                                    neighbor_lookup,
                                    merger_to_metric)

        shuffler.shuffle(new_merger_candidates)

        local_t0 = time.time()
        timeseries_lookup = update_timeseries_lookup(
                              new_merger_candidates,
                              timeseries_lookup,
                              sub_video_lookup,
                              filter_fraction=filter_fraction,
                              n_processors=n_processors)

        logger.info('updated timeseries lookup '
                    f'in {time.time()-local_t0:.2f} seconds')

        local_t0 = time.time()

        new_merger_metrics = get_merger_metric_from_pairs(
                                   new_merger_candidates,
                                   sub_video_lookup,
                                   timeseries_lookup,
                                   filter_fraction,
                                   n_processors)

        logger.info('updated self_corr and calculated metrics in '
                    f'{time.time()-local_t0:.2f} seconds')

        for pair in new_merger_metrics:
            metric = new_merger_metrics[pair]
            if metric >= -1.0*corr_acceptance:
                merger_to_metric[pair] = metric

        keep_going = False
        (recently_merged,
         roi_lookup,
         neighbor_lookup,
         have_been_absorbed) = _do_mergers(merger_to_metric,
                                           roi_lookup,
                                           neighbor_lookup,
                                           have_been_absorbed)
        if len(recently_merged) > 0:
            keep_going = True

        (roi_lookup,
         anomalous_rois) = break_out_anomalous_rois(roi_lookup,
                                                    anomalous_rois,
                                                    anomalous_size)

        (neighbor_lookup,
         sub_video_lookup,
         timeseries_lookup,
         merger_to_metric) = update_lookup_tables(
                                    roi_lookup,
                                    recently_merged,
                                    neighbor_lookup,
                                    sub_video_lookup,
                                    timeseries_lookup,
                                    merger_to_metric)

        logger.info(f'merged {n0} ROIs to {len(roi_lookup)}; '
                    f'{len(anomalous_rois)} anomalous ROIs; '
                    f'after {time.time()-t0:.2f} seconds')

        # make sure we did not lose track of any ROIs
        for roi_id in incoming_rois:
            if roi_id not in have_been_absorbed:
                if roi_id not in roi_lookup:
                    if roi_id not in anomalous_rois:
                        raise RuntimeError(f"lost track of {roi_id}")

    logger.info(f'{len(anomalous_rois)} anomalously large ROIs found '
                f' (size >= {anomalous_size})')
    new_roi_list = list(roi_lookup.values()) + list(anomalous_rois.values())
    return new_roi_list
