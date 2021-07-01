"""
This module contains the code that roi_merging.py uses to
find all pairs of ROIs that need to be considered for merger.
"""
from typing import List, Optional, Tuple, Set, Dict, Union
from itertools import combinations
import numpy as np
import multiprocessing
import multiprocessing.managers
from ophys_etl.modules.segmentation.merge.utils import (
    _winnow_process_list)
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.merge.roi_utils import (
    do_rois_abut)


def _find_merger_candidates(
        roi_id_pair_list: List[Tuple[int, int]],
        roi_lookup: Dict[int, OphysROI],
        pixel_distance: float,
        rois_to_ignore: Optional[Set[int]],
        output_list: Union[list,
                           multiprocessing.managers.ListProxy]) -> None:
    """
    Find all of the abutting ROIs in a list of OphysROIs

    Parameters
    ----------
    roi_id_pair_list: List[Tuple[int, int]]
        Pair of roi_ids to consider for merging

    roi_lookup: Dict[int, OphysROI]
        Maps roi_id to OphysROI

    pixel_distance: float
       The maximum distance from each other two ROIs can be at
       their nearest point and still be considered to abut

    rois_to_ignore: Optional[Set[int]]
       Optional set of ints specifying roi_id of ROIs not to consider
       when looking for pairs. Note: a pair will only be ignored
       if *both* ROIs are in rois_to_ignore. If one of them is not,
       the pair is valid (default: None)

    output_list: Union[list, multiprocessing.managers.ListProxy]
        List where valid merger candidates are stored

    Returns
    -------
    None
        valid merger candidates are stored in output_list
    """
    for roi_id_pair in roi_id_pair_list:
        if roi_id_pair[0] == roi_id_pair[1]:
            continue

        if rois_to_ignore is not None:
            if roi_id_pair[0] in rois_to_ignore:
                if roi_id_pair[1] in rois_to_ignore:
                    continue

        roi0 = roi_lookup[roi_id_pair[0]]
        roi1 = roi_lookup[roi_id_pair[1]]
        if do_rois_abut(roi0, roi1, pixel_distance=pixel_distance):
            output_list.append(roi_id_pair)
    return None


def find_merger_candidates(
        roi_list: List[OphysROI],
        pixel_distance: float,
        rois_to_ignore: Optional[Set[int]] = None,
        n_processors: int = 8) -> List[Tuple[int, int]]:
    """
    Find all the pairs of abutting ROIs in a list of OphysROIs.
    Return a list of tuples like (roi_id_0, roi_id_1) specifying
    the ROIs that abut.

    Parameters
    ----------
    roi_list: List[OphysROI]

    pixel_distance: float
       The maximum distance from each other two ROIs can be at
       their nearest point and still be considered to abut

    rois_to_ignore: Optional[Set[int]]
       Optional set of ints specifying roi_id of ROIs not to consider
       when looking for pairs. Note: a pair will only be ignored
       if *both* ROIs are in rois_to_ignore. If one of them is not,
       the pair is valid (default: None)

    n_processors: int
       Number of cores to use (this function uses multiprocessing since, for
       full fields of view, there can be tens of millions of pairs of
       ROIs to consider)

   Returns
   -------
   output: List[Tuple[int, int]]
       List of tuples of roi_ids specifying pairs of abutting ROIs
    """
    lookup = dict()
    roi_id_list = []
    for roi in roi_list:
        lookup[roi.roi_id] = roi
        roi_id_list.append(roi.roi_id)

    mgr = multiprocessing.Manager()
    output_list = mgr.list()
    n_rois = len(roi_id_list)
    n_pairs = (n_rois)*(n_rois-1)//2
    chunk_size = n_pairs//(n_processors-1)
    chunk_size = max(chunk_size, 1)
    process_list = []
    pair_list = []
    for combo in combinations(roi_id_list, 2):
        pair_list.append(combo)
        if len(pair_list) >= chunk_size:
            args = (pair_list,
                    lookup,
                    pixel_distance,
                    rois_to_ignore,
                    output_list)

            p = multiprocessing.Process(target=_find_merger_candidates,
                                        args=args)
            p.start()
            process_list.append(p)
            pair_list = []
        while len(process_list) > 0 and len(process_list) >= (n_processors-1):
            process_list = _winnow_process_list(process_list)

    if len(pair_list) > 0:

        args = (pair_list,
                lookup,
                pixel_distance,
                rois_to_ignore,
                output_list)

        p = multiprocessing.Process(target=_find_merger_candidates,
                                    args=args)
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    pair_list = [p for p in output_list]
    return pair_list


def create_neighbor_lookup(roi_lookup: Dict[int, OphysROI],
                           n_processors: int) -> Dict[int, Set[int]]:
    """
    Create the initial dict mapping ROI ID to a list of all neighboring
    ROI IDs

    Parameters
    ----------
    roi_lookup: Dict[int, OphysROI]

    n_processors: int
        Number of processors to invoke with multiprocessing

    Returns
    -------
    neighbor_lookup: Dict[int, Set[int]]
        Maps ROI ID to a list of the ROI IDs of all neighboring
        ROIs
    """

    merger_candidates = find_merger_candidates(list(roi_lookup.values()),
                                               np.sqrt(2.0),
                                               rois_to_ignore=None,
                                               n_processors=n_processors)

    # construct a dict mapping an ROI ID to the list of its
    # neighboring ROI IDs (prevents us from having to call
    # find_merger_candidates on every iteration)
    neighbor_lookup = dict()
    for pair in merger_candidates:
        if pair[0] not in neighbor_lookup:
            neighbor_lookup[pair[0]] = set()
        if pair[1] not in neighbor_lookup:
            neighbor_lookup[pair[1]] = set()
        neighbor_lookup[pair[1]].add(pair[0])
        neighbor_lookup[pair[0]].add(pair[1])

    return neighbor_lookup
