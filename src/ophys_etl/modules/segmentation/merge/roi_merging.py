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
    merge.roi_utils import (
        merge_rois,
        do_rois_abut,
        extract_roi_to_ophys_roi,
        ophys_roi_to_extract_roi)

from ophys_etl.modules.segmentation.merge.candidates import (
    find_merger_candidates)

from ophys_etl.modules.segmentation.merge.metric import (
    get_merger_metric_from_pairs)

from ophys_etl.modules.segmentation.merge.characteristic_pixel import (
    update_key_pixel_lookup)

from ophys_etl.modules.segmentation.\
    merge.roi_time_correlation import (
        sub_video_from_roi,
        get_self_correlation)
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

import logging
import time


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def merge_segmentation_rois(uphill_roi: SegmentationROI,
                            downhill_roi: SegmentationROI,
                            new_roi_id: int,
                            new_flux_value: float) -> SegmentationROI:
    """
    Merge two SegmentationROIs, making sure that the ROIs actually
    abut and that the uphill ROI has a larger flux value than the
    downhill ROI (this is a requirement imposed by the way we are
    currently using this method to merge ROIs for cell segmentation)

    Parameters
    ----------
    uphill_roi: SegmentationROI
        The ROI with the larger flux_value

    downhill_roi: SegmentationROI
        The ROI with the smaller flux_value

    new_roi_id: int
        The roi_id to assign to the new ROI

    new_flux_value: float
        The flux value to assign to the new ROI

    Return
    ------
    SegmentationROI

    Raises
    ------
    Runtime error if there is no valid way, always stepping downhill
    in flux_value, to go from uphill_roi.peak to downhill_roi
    """

    new_roi = merge_rois(uphill_roi, downhill_roi, new_roi_id=new_roi_id)
    return SegmentationROI.from_ophys_roi(new_roi,
                                          ancestors=[uphill_roi, downhill_roi],
                                          flux_value=new_flux_value)


def create_segmentation_roi_lookup(raw_roi_list: List[OphysROI],
                                   dx: int = 20) -> Dict[int, SegmentationROI]:
    """
    Create a lookup table mapping roi_id to SegmentationROI.

    The flux_values assigned to each ROI will be median z score of the
    pixels in the ROI in img_data relative to the background of
    non-ROI pixels in a neighborhood centered on the ROI.

    Parameters
    ----------
    raw_roi_list: List[OphysROI]

    dx: int
        The number of pixels above, below, to the left, and to the right of
        the ROI used when constructing the neighborhood of pixels used to
        calculate flux_value

    Returns
    -------
    lookup: Dict[int, SegmentationROI]
        A dict allowing you to lookup the SegmentationROI based on its
        roi_id
    """
    lookup = {}
    for roi in raw_roi_list:
        new_roi = SegmentationROI.from_ophys_roi(roi,
                                                 ancestors=None,
                                                 flux_value=0.0)

        if new_roi.roi_id in lookup:
            msg = f'{new_roi.roi_id} duplicated in '
            msg += 'segmentation_roi_lookup'
            raise RuntimeError(msg)

        lookup[new_roi.roi_id] = new_roi

    return lookup


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


def do_roi_merger(
      raw_roi_list: List[OphysROI],
      video_data: np.ndarray,
      n_processors: int,
      corr_acceptance: float,
      filter_fraction: float = 0.2) -> List[SegmentationROI]:
    """
    Merge ROIs based on a static image.

    Parameters
    ----------
    raw_roi_list: List[OphysROI]

    video_data: np.ndarray
        (ntime, nrows, ncols)

    n_processors: int
        The number of processors to invoke with multiprocessing
        (only used when comparing all pairs of ROIs to find neighbors)

    corr_acceptance: float
        The pixel time correlation threshold for accepting a
        merger (see "Notes")

    filter_fraction: float
        Fraction of timesteps to keep when correlating pixels
        in sub_videos (default=0.2)

    Returns
    -------
    List[SegmentationROI]
        List of ROIs after merger. ROIs will have been cast
        to SegmentationROIs, but they have the same spatial
        information and API as OphysROIs

    Notes
    -----
    This algorithm works as follows:
    1) Find all pairs of ROIs that are neighbors (in this case, being
    a neighbor means physically abutting one another)

    2) Loop over all pairs of ROIs. For each pair, assess the validity
    of the merger by

        2a) Correlate all of the pixels in roi0 against the brightest
        pixel in roi0 using the brightest filter_fraction of time steps.
        Use these correlations to construct a Gaussian distribution.

        2b) Correlate all of the pixels in roi1 against the brightest
        pixel in roi0 using the same timesteps.

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

    # create a lookup table of SegmentationROIs
    t0 = time.time()
    roi_lookup = create_segmentation_roi_lookup(raw_roi_list,
                                                dx=20)
    logger.info(f'created roi lookup in {time.time()-t0:.2f} seconds')

    t0 = time.time()
    logger.info('starting merger')
    keep_going = True
    have_been_merged = set()
    i_pass = -1
    incoming_rois = list(roi_lookup.keys())
    valid_roi_id = set(roi_lookup.keys())

    shuffler = np.random.RandomState(11723412)

    merger_candidates = find_merger_candidates(list(roi_lookup.values()),
                                               np.sqrt(2.0),
                                               rois_to_ignore=None,
                                               n_processors=n_processors)

    neighbor_lookup = {}
    for pair in merger_candidates:
        if pair[0] not in neighbor_lookup:
            neighbor_lookup[pair[0]] = set()
        if pair[1] not in neighbor_lookup:
            neighbor_lookup[pair[1]] = set()
        neighbor_lookup[pair[1]].add(pair[0])
        neighbor_lookup[pair[0]].add(pair[1])

    logger.info('found global merger candidates in '
                f'{time.time()-t0:.2f} seconds')

    # global lookup tables for per-ROI sub-videos and mappings
    # between potential merger pairs and merger metrics
    sub_video_lookup = {}
    merger_to_metric = {}
    pixel_lookup = {}
    self_corr_lookup = {}

    anomalous_rois = {}

    while keep_going:
        logger.info(f'starting iteration; {len(roi_lookup)} valid ROIs; '
                    f'{len(anomalous_rois)} anomalous ROIs')
        keep_going = False
        t0_pass = time.time()
        n0 = len(roi_lookup)
        i_pass += 1

        # fill in sub-video lookup for ROIs that changed
        # during the last iteration
        for roi_id in roi_lookup:
            if roi_id in sub_video_lookup:
                continue
            roi = roi_lookup[roi_id]
            sub_video_lookup[roi_id] = sub_video_from_roi(roi,
                                                          video_data)
        logger.info(f'got sub video lookup in {time.time()-t0_pass:.2f}')

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

        shuffler.shuffle(merger_candidates)

        logger.info(f'found {len(merger_candidates)} merger candidates '
                    f'in {time.time()-t0_pass:.2f} seconds')

        pixel_lookup = update_key_pixel_lookup(
                          merger_candidates,
                          roi_lookup,
                          pixel_lookup,
                          sub_video_lookup,
                          n_processors)


        logger.info('updated pixel lookup '
                    f'in {time.time()-t0_pass:.2f} seconds')

        self_corr_lookup = update_self_correlation(
                               merger_candidates,
                               sub_video_lookup,
                               pixel_lookup,
                               filter_fraction,
                               self_corr_lookup,
                               n_processors)

        logger.info('updated self_corr lookup '
                    f'in {time.time()-t0_pass:.2f} seconds')

        new_merger_metrics = get_merger_metric_from_pairs(
                                   merger_candidates,
                                   sub_video_lookup,
                                   pixel_lookup,
                                   self_corr_lookup,
                                   filter_fraction,
                                   n_processors)

        logger.info(f'calculated metrics after {time.time()-t0_pass:.2f}')

        for potential_merger in new_merger_metrics:
            pair = (potential_merger[0], potential_merger[1])
            metric = new_merger_metrics[potential_merger]
            if metric >= -1.0*corr_acceptance:
                merger_to_metric[pair] = metric

        potential_mergers = []
        merger_metrics = []
        for pair in merger_to_metric:
            potential_mergers.append(pair)
            merger_metrics.append(merger_to_metric[pair])

        potential_mergers = np.array(potential_mergers)
        merger_metrics = np.array(merger_metrics)
        sorted_indices = np.argsort(-1*merger_metrics)
        potential_mergers = potential_mergers[sorted_indices]

        recently_merged = set()
        wait_for_it = set()
        larger = []
        smaller = []

        area_lookup = {}
        for roi_id in roi_lookup:
            area_lookup[roi_id] = roi_lookup[roi_id].area

        for merger in potential_mergers:
            roi_id_0 = merger[0]
            roi_id_1 = merger[1]
            go_ahead = True

            if roi_id_0 not in valid_roi_id:
                go_ahead = False
            if roi_id_1 not in valid_roi_id:
                go_ahead = False

            if go_ahead:
                if roi_id_0 in recently_merged:
                    if roi_lookup[roi_id_0].area > 1.1*area_lookup[roi_id_0]:
                        go_ahead = False
                if roi_id_1 in recently_merged:
                    if roi_lookup[roi_id_1].area > 1.1*area_lookup[roi_id_1]:
                        go_ahead = False

            if go_ahead:
                if roi_id_0 in wait_for_it:
                    go_ahead = False
                if roi_id_1 in wait_for_it:
                    go_ahead = False

            if not go_ahead:
                wait_for_it.add(roi_id_0)
                wait_for_it.add(roi_id_1)
                continue

            if roi_lookup[roi_id_0].area > roi_lookup[roi_id_1].area:
                seed_id = roi_id_0
                child_id = roi_id_1
            else:
                seed_id = roi_id_1
                child_id = roi_id_0

            seed_roi = roi_lookup[seed_id]
            child_roi = roi_lookup[child_id]
            larger.append(seed_roi.area)
            smaller.append(child_roi.area)
            keep_going = True
            new_roi = merge_segmentation_rois(seed_roi,
                                              child_roi,
                                              seed_roi.roi_id,
                                              seed_roi.flux_value)
            roi_lookup.pop(child_roi.roi_id)
            valid_roi_id.remove(child_roi.roi_id)
            have_been_merged.add(child_roi.roi_id)
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

        # break out any ROIs that are too large
        k_list = list(roi_lookup.keys())
        for roi_id in k_list:
            if roi_lookup[roi_id].area >= anomalous_size:
                roi = roi_lookup.pop(roi_id)
                roi.valid_roi = False
                anomalous_rois[roi_id] = roi

                valid_roi_id.remove(roi_id)
                neighbor_lookup.pop(roi_id)
                sub_video_lookup.pop(roi_id)
                self_corr_lookup.pop(roi_id)
                pixel_lookup.pop(roi_id)

        # remove an obsolete ROI IDs from neighbor lookup
        for roi_id in neighbor_lookup:
            new_set = neighbor_lookup[roi_id].intersection(valid_roi_id)
            neighbor_lookup[roi_id] = new_set

        # remove ROIs that have changed from sub_video and
        # merger metric lookup tables
        for roi_id in recently_merged:
            if roi_id in sub_video_lookup:
                sub_video_lookup.pop(roi_id)
                self_corr_lookup.pop(roi_id)

        # remove non-existent ROIs and ROIs whose areas
        # have significantly changed from pixel_lookup
        k_list = list(pixel_lookup.keys())
        for roi_id in k_list:
            pop_it = False
            if roi_id not in valid_roi_id:
                pop_it = True
            elif roi_lookup[roi_id].area > 1.05*pixel_lookup[roi_id]['area']:
                pop_it = True

            if pop_it:
                pixel_lookup.pop(roi_id)

        merger_keys = list(merger_to_metric.keys())
        for pair in merger_keys:
            if pair[0] in recently_merged or pair[1] in recently_merged:
                merger_to_metric.pop(pair)

        logger.info(f'done processing after {time.time()-t0_pass:.2f}')

        logger.info(f'merged {n0} ROIs to {len(roi_lookup)} '
                    f'after {time.time()-t0:.2f} seconds')

        if len(larger) > 0:
            larger = np.array(larger)
            smaller = np.array(smaller)
            ratio = smaller/larger
            quartiles = np.quantile(ratio, [0.25, 0.5, 0.75])
            logger.info(f'{len(ratio)} mergers; ratios '
                        f'{quartiles[0]:.2f} {quartiles[1]:.2f} '
                        f'{quartiles[2]:.2f} {ratio.max() :.2f}')

            quartiles = np.quantile(larger, [0.25, 0.5, 0.75])
            logger.info('larger '
                        f'{quartiles[0]:.2f} {quartiles[1]:.2f} '
                        f'{quartiles[2]:.2f} {larger.max() :.2f}')

        # make sure we did not lose track of any ROIs
        for roi_id in incoming_rois:
            if roi_id not in have_been_merged:
                if roi_id not in roi_lookup:
                    if roi_id not in anomalous_rois:
                        raise RuntimeError(f"lost track of {roi_id}")

    logger.info(f'{len(anomalous_rois)} anomalously large ROIs found '
                f' (size >= {anomalous_size})')
    # loop over the original list of roi_ids, copying
    # any ROIs that were not merged into the output list
    new_roi_list = list(roi_lookup.values()) + list(anomalous_rois.values())
    return new_roi_list
