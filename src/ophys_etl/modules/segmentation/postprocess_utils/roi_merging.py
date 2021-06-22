from typing import List, Optional, Tuple, Dict, Set
import multiprocessing
import multiprocessing.managers
from scipy.spatial.distance import cdist
import numpy as np
import copy
import pathlib
from ophys_etl.modules.segmentation.postprocess_utils.roi_types import (
    SegmentationROI)
from ophys_etl.modules.segmentation.\
    postprocess_utils.roi_time_correlation import (
        validate_merger_corr,
        sub_video_from_roi)
from ophys_etl.modules.decrosstalk.ophys_plane import get_roi_pixels
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.types import ExtractROI

import logging
import json
import time


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def _winnow_process_list(
    process_list: List[multiprocessing.Process]) \
        -> List[multiprocessing.Process]:
    """
    Utility that loops over a list of multiprocessing.Processes and
    pops any that have completed. Returns the new, truncated list of
    multiprocessing.Processes
    """

    to_pop = []
    for ii in range(len(process_list)-1, -1, -1):
        if process_list[ii].exitcode is not None:
            to_pop.append(ii)
    for ii in to_pop:
        process_list.pop(ii)
    return process_list


def extract_roi_to_ophys_roi(roi: ExtractROI) -> OphysROI:
    """
    Convert an ExtractROI to an equivalent OphysROI

    Parameters
    ----------
    ExtractROI

    Returns
    -------
    OphysROI
    """
    new_roi = OphysROI(x0=roi['x'],
                       y0=roi['y'],
                       width=roi['width'],
                       height=roi['height'],
                       mask_matrix=roi['mask'],
                       roi_id=roi['id'],
                       valid_roi=True)

    return new_roi


def ophys_roi_to_extract_roi(roi: OphysROI) -> ExtractROI:
    """
    Convert at OphysROI to an equivalent ExtractROI

    Parameters
    ----------
    OphysROI

    Returns
    -------
    ExtractROI
    """
    mask = []
    for roi_row in roi.mask_matrix:
        row = []
        for el in roi_row:
            if el:
                row.append(True)
            else:
                row.append(False)
        mask.append(row)

    new_roi = ExtractROI(x=roi.x0,
                         y=roi.y0,
                         width=roi.width,
                         height=roi.height,
                         mask=mask,
                         valid_roi=True,
                         id=roi.roi_id)
    return new_roi


def _do_rois_abut(array_0: np.ndarray,
                  array_1: np.ndarray,
                  dpix: float = np.sqrt(2)) -> bool:
    """
    Method that does the work behind user-facing do_rois_abut.

    This method takes in two arrays of pixel coordinates
    calculates the distance between every pair of pixels
    across the two arrays. If the minimum distance is less
    than or equal dpix, it returns True. If not, it return False.

    Parameters
    ----------
    array_0: np.ndarray
        Array of the first set of pixels. Shape is (npix0, 2).
        array_0[:, 0] are the row coodinates of the pixels
        in array_0. array_0[:, 1] are the column coordinates.

    array_1: np.ndarray
        Same as array_0 for the second array of pixels

    dpix: float
        Maximum distance two arrays can be from each other
        at their closest point and still be considered
        to abut (default: sqrt(2)).

    Return
    ------
    boolean
    """
    distances = cdist(array_0, array_1, metric='euclidean')
    if distances.min() <= dpix:
        return True
    return False


def _get_pixel_array(roi: OphysROI) -> np.ndarray:
    """
    get Nx2 array of pixels (in global coordinates)
    that are in the ROI

    Parameters
    ----------
    OphysROI

    Returns
    -------
    np.ndarray
    """
    mask = roi.mask_matrix
    n_bdry = mask.sum()
    roi_array = -1*np.ones((n_bdry, 2), dtype=int)
    i_pix = 0
    for ir in range(roi.height):
        row = ir+roi.y0
        for ic in range(roi.width):
            col = ic+roi.x0
            if not mask[ir, ic]:
                continue

            roi_array[i_pix, 0] = row
            roi_array[i_pix, 1] = col
            i_pix += 1

    if roi_array.min() < 0:
        raise RuntimeError("did not assign all boundary pixels")

    return roi_array


def do_rois_abut(roi0: OphysROI,
                 roi1: OphysROI,
                 dpix: float = np.sqrt(2)) -> bool:
    """
    Returns True if ROIs are within dpix of each other at any point.

    Parameters
    ----------
    roi0: OphysROI

    roi1: OphysROI

    dpix: float
        The maximum distance from each other the ROIs can be at
        their closest point and still be considered to abut.
        (Default: np.sqrt(2))

    Returns
    -------
    boolean

    Notes
    -----
    dpix is such that if two boundaries are next to each other,
    that corresponds to dpix=1; dpix=2 corresponds to 1 blank pixel
    between ROIs
    """
    array_0 = _get_pixel_array(roi0)
    array_1 = _get_pixel_array(roi1)

    return _do_rois_abut(array_0,
                         array_1,
                         dpix=dpix)


def merge_rois(roi0: OphysROI,
               roi1: OphysROI,
               new_roi_id: int) -> OphysROI:
    """
    Merge two OphysROIs into one OphysROI whose
    mask is the union of the two input masks

    Parameters
    ----------
    roi0: OphysROI

    roi1: OphysROI

    new_roi_id: int
        The roi_id to assign to the output ROI

    Returns
    -------
    OphysROI
    """

    xmin0 = roi0.x0
    xmax0 = roi0.x0+roi0.width
    ymin0 = roi0.y0
    ymax0 = roi0.y0+roi0.height
    xmin1 = roi1.x0
    xmax1 = roi1.x0+roi1.width
    ymin1 = roi1.y0
    ymax1 = roi1.y0+roi1.height

    xmin = min(xmin0, xmin1)
    xmax = max(xmax0, xmax1)
    ymin = min(ymin0, ymin1)
    ymax = max(ymax0, ymax1)

    width = xmax-xmin
    height = ymax-ymin

    mask = np.zeros((height, width), dtype=bool)

    pixel_dict = get_roi_pixels([roi0, roi1])
    for roi_id in pixel_dict:
        roi_mask = pixel_dict[roi_id]
        for pixel in roi_mask:
            mask[pixel[1]-ymin, pixel[0]-xmin] = True

    new_roi = OphysROI(x0=xmin,
                       y0=ymin,
                       width=width,
                       height=height,
                       mask_matrix=mask,
                       roi_id=new_roi_id,
                       valid_roi=True)

    return new_roi


def chunk_size_from_processors(n_elements: int,
                               n_cores: int,
                               min_chunk: int,
                               denom_factor: int = 4) -> int:
    """
    Given a number of data elements that need to be
    processed and a number of available processors,
    try to find a good chunk size so that the processors
    are always busy.

    Parameters
    ----------
    n_elements: int
        The number of data elements that you are trying
        to chunk

    n_cores: int
        The number of available cores

    min_chunk: int
        Minimum acceptable chunk_size

    denom_factor: int
        number of chunks that should ultimately be
        sent to each core (default=4)

    Returns
    -------
    chunk_size: int
    """
    chunk_size = n_elements//(denom_factor*n_cores-1)
    if chunk_size < min_chunk:
        chunk_size = min_chunk
    return chunk_size


def _find_merger_candidates(
        roi_pair_list: List[Tuple[OphysROI, OphysROI]],
        dpix: float,
        output_list: multiprocessing.managers.ListProxy) -> None:
    """
    Find all of the abutting ROIs in a list of OphysROIs

    Parameters
    ----------
    roi_pair_list: List[Tuple[OphysROI, OphysROI]]
        A list of tuples of OphysROIs that need to be tested
        to see if they abut

    dpix: float
       The maximum distance from each other two ROIs can be at
       their nearest point and still be considered to abut

    output_list: multiprocessing.managers.ListProxy
        List where tuples of IDs of abutting ROIs will be written

    Returns
    -------
    None
        Appends tuples of (roi_id_0, roi_id_1) corresponding
        to abutting ROIs into output_list
    """
    local = []
    for pair in roi_pair_list:
        if do_rois_abut(pair[0], pair[1], dpix=dpix):
            local.append((pair[0].roi_id, pair[1].roi_id))
    for pair in local:
        output_list.append(pair)
    return None


def find_merger_candidates(roi_list: List[OphysROI],
                           dpix: float,
                           rois_to_ignore: Optional[set] = None,
                           n_processors: int = 8) -> List[Tuple[int, int]]:
    """
    Find all the pairs of abutting ROIs in a list of OphysROIs.
    Return a list of tuples like (roi_id_0, roi_id_1) specifying
    the ROIs that abut.

    Parameters
    ----------
    roi_list: List[OphysROI]

    dpix: float
       The maximum distance from each other two ROIs can be at
       their nearest point and still be considered to abut

    rois_to_ignore: Optional[set]
       Optional set of ints specifying roi_id of ROIs not to consider
       when looking for pairs. Note: a pair will only be ignored
       if *both* ROIs are in rois_to_ignore. If one of them is not,
       the pair is valid (default: None)

    n_processors: int
       Number of cores to use (this method uses multiprocessing since, for
       full fields of view, there can be tens of millions of pairs of
       ROIs to consider)

   Returns
   -------
   output: List[Tuple[int, int]]
       List of tuples of roi_ids specifying pairs of abutting ROIs
    """
    mgr = multiprocessing.Manager()
    output_list = mgr.list()

    n_rois = len(roi_list)

    process_list = []

    n_pairs = n_rois*(n_rois-1)//2
    d_pairs = chunk_size_from_processors(n_pairs, n_processors, 100)

    subset = []
    for i0 in range(n_rois):
        roi0 = roi_list[i0]
        for i1 in range(i0+1, n_rois, 1):
            roi1 = roi_list[i1]
            if rois_to_ignore is None:
                subset.append((roi0, roi1))
            else:
                if (roi0.roi_id not in rois_to_ignore
                        or roi1.roi_id not in rois_to_ignore):

                    subset.append((roi0, roi1))
            if len(subset) >= d_pairs:
                args = (copy.deepcopy(subset), dpix, output_list)
                p = multiprocessing.Process(target=_find_merger_candidates,
                                            args=args)
                p.start()
                process_list.append(p)
                subset = []
            while (len(process_list) > 0
                   and len(process_list) >= (n_processors-1)):
                process_list = _winnow_process_list(process_list)

    if len(subset) > 0:
        args = (subset, dpix, output_list)
        p = multiprocessing.Process(target=_find_merger_candidates,
                                    args=args)
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    pair_list = [pair for pair in output_list]
    return pair_list


def get_inactive_mask(img_data_shape: Tuple[int, int],
                      roi_list: List[OphysROI]) -> np.ndarray:
    """
    Return a pixel mask that is marked False for all
    pixels that are part of an ROI, and True for all
    other pixels

    Parameters
    ----------
    img_data_shape: Tuple[int, int]
        Shape of the full field of view

    roi_list: List[OphysROI]

    Returns
    -------
    inactive_pixel_mask: np.ndarray
    """
    full_mask = np.zeros(img_data_shape, dtype=bool)
    for roi in roi_list:
        mask = roi.mask_matrix
        region = full_mask[roi.y0:roi.y0+roi.height,
                           roi.x0:roi.x0+roi.width]
        full_mask[roi.y0:roi.y0+roi.height,
                  roi.x0:roi.x0+roi.width] = np.logical_or(mask, region)

    return np.logical_not(full_mask)


def get_inactive_distribution(img_data: np.ndarray,
                              roi: OphysROI,
                              inactive_mask: np.ndarray,
                              dx: int = 10) -> Tuple[float, float]:
    """
    Given an ROI, an array of image data, and a mask marked True at
    all of the non-ROI pixels, return the mean and standard deviation
    of the inactive pixels in a neighborhood about the ROI.

    Parameters
    ----------
    img_data: np.ndarray
        The full field of view image data

    roi: OphysROI

    inactive_mask: np.ndarray
        An array of booleans the same shape as img_data.
        Marked True for any pixel not in an ROI; False
        for all other pixels.

    dx: int
        Number of pixels to either side of the ROI to use
        when constructing the neighborhood.

    Returns
    -------
    mu: float
        Mean of the inactive pixels in the neighborhood

    std: float
        Standard deviation of the inactive pixels in the
        neighborhood
    """

    xmin = max(0, roi.x0-dx)
    ymin = max(0, roi.y0-dx)
    xmax = xmin+roi.width+2*dx
    ymax = ymin+roi.height+2*dx

    if xmax > img_data.shape[1]:
        xmax = img_data.shape[1]
        xmin = max(0, xmax-roi.width-2*dx)
    if ymax > img_data.shape[1]:
        ymin = max(0, ymax-roi.height-2*dx)

    neighborhood = img_data[ymin:ymax, xmin:xmax]
    mask = inactive_mask[ymin:ymax, xmin:xmax]
    inactive_pixels = neighborhood[mask].flatten()
    mu = np.mean(inactive_pixels)
    if len(inactive_pixels) < 2:
        std = 0.0
    else:
        std = np.std(inactive_pixels, ddof=1)

    return mu, std


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
                                   img_data: np.ndarray,
                                   dx: int = 20) -> Dict[int, SegmentationROI]:
    """
    Create a lookup table mapping roi_id to SegmentationROI.

    The flux_values assigned to each ROI will be median z score of the
    pixels in the ROI in img_data relative to the background of
    non-ROI pixels in a neighborhood centered on the ROI.

    Parameters
    ----------
    raw_roi_list: List[OphysROI]

    img_data: np.ndarray
        The image data used to calculate the flux_value of each SegmentationROI

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
    inactive_mask = get_inactive_mask(img_data.shape, raw_roi_list)
    for roi in raw_roi_list:
        mu, sigma = get_inactive_distribution(img_data,
                                              roi,
                                              inactive_mask,
                                              dx=dx)
        mask = roi.mask_matrix
        xmin = roi.x0
        xmax = xmin + roi.width
        ymin = roi.y0
        ymax = ymin + roi.height
        roi_pixels = img_data[ymin:ymax, xmin:xmax][mask].flatten()
        n_sigma = np.median((roi_pixels-mu)/sigma)
        new_roi = SegmentationROI.from_ophys_roi(roi,
                                                 ancestors=None,
                                                 flux_value=n_sigma)

        if new_roi.roi_id in lookup:
            msg = f'{new_roi.roi_id} duplicated in '
            msg += 'segmentation_roi_lookup'
            raise RuntimeError(msg)

        lookup[new_roi.roi_id] = new_roi

    return lookup


def _validate_mergers(input_pair_list,
                      roi_lookup,
                      video_lookup,
                      img_data,
                      filter_fraction,
                      corr_acceptance,
                      output_pair_list):

    for pair in input_pair_list:
        roi0 = roi_lookup[pair[0]]
        roi1 = roi_lookup[pair[1]]
        valid = validate_merger_corr(roi0,
                                     roi1,
                                     video_lookup,
                                     img_data,
                                     filter_fraction=filter_fraction,
                                     acceptance=corr_acceptance)
        if valid[0]:
            output_pair_list.append((pair[0], pair[1], valid[1]))


def do_roi_merger(
      raw_roi_list: List[OphysROI],
      img_data: np.ndarray,
      video_data: np.ndarray,
      n_processors: int,
      corr_acceptance: float,
      filter_fraction = 0.2) -> List[SegmentationROI]:
    """
    Merge ROIs based on a static image.

    Parameters
    ----------
    raw_roi_list: List[OphysROI]

    img_data: np.ndarray
        The static image used to guide merging

    video_data: np.ndarray
        (ntime, nrows, ncols)

    n_processors: int
        The number of processors to invoke with multiprocessing
        (only used when comparing all pairs of ROIs to find neighbors)

    corr_acceptance: float
        The pixel time correlation threshold for accepting a
        merger (see "Notes")

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

    2) Assign to each ROI a uniform brightness that is the median
    of its z-score relative to the local background of non-ROI pixels.

    3) Identify all ROIs that are brighter then each of their neighbors.
    These are the seeds of the merging process.

    4) Iteratively merge ROIs by following paths that are "downhill"
    in brightness (i.e. do not merge an ROI into a group unless there
    is a path from the peak ROI to the new ROI that is monotonically
    descending in brightness).

    4a) When considering a merger, select the brightest pixel in the
    uphill ROI. Treating every pixel in the ROI as a time series,
    correlate each pixel in the uphill ROI with the brightest pixel,
    using only the brightest 80% of timesteps (for the brightest pixel).
    Use this set of correlations to construct a Gaussian distribution.
    Correlate every pixel in the downhill ROI against the brightest
    pixel in the uphill ROI. Assess the z-scores of these cross
    correlations relative to the distribution of the self-correlation
    of pixels in the uphill ROI. Only accept the merger if the
    median z-score is greater than -1*corr_acceptance (i.e. if the
    pixels in the downhill ROI are, on average, more correlated
    with the brightest pixel in the uphill ROI than some threshold
    specified by corr_acceptance).

    5) Continue until there are no more mergers

    As implemented, if an ROI is between two peaks, it will ultimately
    get merged with the peak that has the fewest intervening ROIs. Future
    development should address this ambiguity more rigorously.
    """

    # create a lookup table of SegmentationROIs
    t0 = time.time()
    roi_lookup = create_segmentation_roi_lookup(raw_roi_list,
                                                img_data,
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


    logger.info(f'found global merger candidates in {time.time()-t0:.2f} seconds')

    # global lookup tables for per-ROI sub-videos and mappings
    # between potential merger pairs and merger metrics
    sub_video_lookup = {}
    merger_to_metric = {}

    while keep_going:
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

        n_pairs = len(merger_candidates)
        chunk_size = chunk_size_from_processors(n_pairs,
                                                n_processors,
                                                1,
                                                2)
        mgr = multiprocessing.Manager()
        output_list = mgr.list()
        process_list = []
        for i0 in range(0, n_pairs, chunk_size):
            chunk = merger_candidates[i0: i0+chunk_size]
            args = (chunk,
                    roi_lookup,
                    sub_video_lookup,
                    img_data,
                    filter_fraction,
                    corr_acceptance,
                    output_list)

            process = multiprocessing.Process(target=_validate_mergers,
                                              args=args)

            process.start()
            process_list.append(process)
            while len(process_list) > 0 and len(process_list) > (n_processors-1):
                process_list = _winnow_process_list(process_list)


        for p in process_list:
            p.join()

        for potential_merger in output_list:
            pair = (potential_merger[0], potential_merger[1])
            metric = potential_merger[2]
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
        for merger in potential_mergers:
            roi_id_0 = merger[0]
            roi_id_1 = merger[1]
            if roi_id_0 not in valid_roi_id:
                continue
            if roi_id_1 not in valid_roi_id:
                continue
            if roi_id_0 in recently_merged:
                continue
            if roi_id_1 in recently_merged:
                continue

            if roi_lookup[roi_id_0].peak.flux_value > roi_lookup[roi_id_1].peak.flux_value:
                seed_id = roi_id_0
                child_id = roi_id_1
            else:
                seed_id = roi_id_1
                child_id = roi_id_0

            seed_roi = roi_lookup[seed_id]
            child_roi = roi_lookup[child_id]
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

        # remove an obsolete ROI IDs from neighbor lookup
        for roi_id in neighbor_lookup:
            new_set = neighbor_lookup[roi_id].intersection(valid_roi_id)
            neighbor_lookup[roi_id] = new_set

        # remove ROIs that have changed from sub_video and
        # merger metric lookup tables
        for roi_id in recently_merged:
            if roi_id in sub_video_lookup:
                sub_video_lookup.pop(roi_id)

        merger_keys = list(merger_to_metric.keys())
        for pair in merger_keys:
            if pair[0] in recently_merged or pair[1] in recently_merged:
                merger_to_metric.pop(pair)

        logger.info(f'done processing after {time.time()-t0_pass:.2f}')

        logger.info(f'merged {n0} ROIs to {len(roi_lookup)} '
                    f'after {time.time()-t0:.2f} seconds')

        # make sure we did not lose track of any ROIs
        for roi_id in incoming_rois:
            if roi_id not in have_been_merged:
                if roi_id not in roi_lookup:
                    raise RuntimeError(f"lost track of {roi_id}")

    # loop over the original list of roi_ids, copying
    # any ROIs that were not merged into the output list
    new_roi_list = list(roi_lookup.values())
    return new_roi_list
