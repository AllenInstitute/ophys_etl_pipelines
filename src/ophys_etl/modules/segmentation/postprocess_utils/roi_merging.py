from __future__ import annotations
from typing import List, Optional, Tuple
import multiprocessing
import multiprocessing.managers
from scipy.spatial.distance import cdist
import numpy as np
import copy
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
    process_list: List[multiprocessing.Process]) -> List[multiprocessing.Process]:
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
            col =ic+roi.x0
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


def _find_merger_candidates(roi_pair_list: List[Tuple[OphysROI, OphysROI]],
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
                           rois_to_ignore: Optional[set]=None,
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
                if roi0.roi_id not in rois_to_ignore or roi1.roi_id not in rois_to_ignore:
                    subset.append((roi0, roi1))
            if len(subset) >= d_pairs:
                args = (copy.deepcopy(subset), dpix, output_list)
                p = multiprocessing.Process(target=_find_merger_candidates,
                                            args=args)
                p.start()
                process_list.append(p)
                subset = []
            while len(process_list) > 0 and len(process_list) >= (n_processors-1):
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


def get_inactive_dist(img_data: np.ndarray,
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



class SegmentationROI(OphysROI):
    """
    A class that expands on the functionality provided by
    OphysROI. Specifically, it carries with it a flux_value
    and a list of ancestor ROIs (in the event that this ROI
    was created by merging some subset of raw ROIs).

    The flux value really only has meaning in the case of ROIs
    with no ancestors.

    The ancestors are all ROIs with no ancestors themselves.

    These properties are used when assessing a merger between
    ROIs to make sure it does not involve the flux value
    going "downhill" and then "uphill" again.

    Parameters
    ----------
    roi_id: int

    x0: int
        Defines the starting x pixel of the mask_array

    y0: int
        Defines the starting y pixel of the mask_array

    width: int
        Defines the width of the mask_array

    height: int
        Definings the width of the mask_array

    valid_roi: bool
        Indicate the validity of the ROI

    mask_matrix: list
        a list of lists of booleans defining the pixels
        that are a part of the ROI

    flux_value: float
        The scalar flux valuea associated with this ROI (default=0)

    ancestors: Optional[List[SegmentationROI]]
        A list of the SegmentationROIs from which this ROI
        was assembled (if relevant). Default=None
    """

    def __init__(self,
                 roi_id: int,
                 x0: int,
                 y0: int,
                 width: int,
                 height: int,
                 valid_roi: bool,
                 mask_matrix: list,
                 flux_value: float = 0.0,
                 ancestors: Optional[List[SegmentationROI]] = None):

        self.flux_value = flux_value

        self.ancestors = []
        if ancestors is not None:
            self.ancestors = []
            for roi in ancestors:
                if not isinstance(roi, SegmentationROI):
                    msg = 'ancestors must be of class SegmentationROI; '
                    msg += f'these ancestors include a {type(sub_roi)}'
                    raise RuntimeError(msg)

                if len(roi.ancestors) == 0:
                    self.ancestors.append(roi)
                else:
                    for sub_roi in roi.ancestors:
                        if not isinstance(sub_roi, SegmentationROI):
                            msg = 'ancestors must be of class SegmentationROI; '
                            msg += f'these ancestors include a {type(sub_roi)}'
                            raise RuntimeError(msg)
                        self.ancestors.append(sub_roi)

        # verify that ancestors have unique ID values
        # (this will be necessary when assessing merger candidates)
        if len(self.ancestors) > 0:
            ancestor_id = set([a.roi_id for a in self.ancestors])
            if len(ancestor_id) != len(self.ancestors):
                msg = 'ancestors do not have unique IDs! '
                msg += f'{len(self.ancestors)} ancestors; '
                msg += f'{len(ancestor_id)} IDs; '
                id_list = list(ancestor_id)
                id_list.sort()
                msg += f'{id_list}'
                raise RuntimeError(msg)

        self._ancestor_lookup = self._create_ancestor_lookup()

        super().__init__(x0=x0, y0=y0,
                         height=height, width=width,
                         valid_roi=valid_roi,
                         mask_matrix=mask_matrix,
                         roi_id=roi_id)

    def _create_ancestor_lookup(self) -> dict:
        """
        Create a lookup table mapping roi_id to SegmentationROI
        """
        lookup = {}
        for a in self.ancestors:
            lookup[a.roi_id] = a
        return lookup

    @classmethod
    def from_ophys_roi(cls,
                       input_roi: OphysROI,
                       ancestors: Optional[list] = None,
                       flux_value: float = 0.0) -> SegmentationROI:
        """
        Create a SegmentationROI from an OphysROI

        Parameters
        ----------
        input_roi: OphysROI

        ancestors: Optional[list]
            List of ancestor SegmentationROIs to assign to the new
            SegmentationROI (default: None)

        flux_value: float
            flux value to assign to the new SegmentationROI (default: 0.0)

        Returns
        -------
        SegmentationROI
        """

        return cls(x0=input_roi.x0,
                   y0=input_roi.y0,
                   width=input_roi.width,
                   height=input_roi.height,
                   valid_roi=input_roi.valid_roi,
                   mask_matrix=input_roi.mask_matrix,
                   roi_id=input_roi.roi_id,
                   ancestors=ancestors,
                   flux_value=flux_value)

    @property
    def peak(self) -> SegmentationROI:
        """
        Return the ancestor with the highest flux_value
        (this is the 'peak' of the group of ROIs that have been
        merged to create this ROI). If there are no ancestors, return
        self.
        """
        if len(self.ancestors) == 0:
            return self
        peak_val = None
        peak_roi = None
        for roi in self.ancestors:
            if peak_val is None or roi.flux_value > peak_val:
                peak_roi = roi
                peak_val = roi.flux_value
        return peak_roi

    def get_ancestor(self, roi_id: int) -> SegmentationROI:
        """
        Return the ancestor with the specified roi_id.

        Note, if this SegmentationROI has the same roi_id as one
        of its ancestors, its ancestor will be returned. Otherwise,
        if you specify this ROI's roi_id, you will get self back.
        """
        if roi_id in self._ancestor_lookup:
            return self._ancestor_lookup[roi_id]
        if roi_id != self.roi_id:
            id_list = list(self._ancestor_lookup.keys())
            id_list.append(self.roi_id)
            id_list.sort()
            raise RuntimeError(f"cannot get ancestor {roi_id}; "
                               f"valid ancestors: {id_list}")
        return self


def merge_segmentation_rois(roi0, roi1, new_roi_id, flux_value):

    has_valid_step = False
    if len(roi0.ancestors)>0:
        for a in roi0.ancestors:
            if do_rois_abut(a, roi1, dpix=np.sqrt(2)):
                if a.flux_value >= (roi1.flux_value+0.001):
                    has_valid_step = True

        abut = do_rois_abut(roi1, roi0, dpix=np.sqrt(2))
        assert has_valid_step

    new_roi = merge_rois(roi0, roi1, new_roi_id=new_roi_id)
    return SegmentationROI.from_ophys_roi(new_roi,
                                          ancestors=[roi0, roi1],
                                          flux_value=flux_value)


def create_segmentation_roi_lookup(raw_roi_list,
                                   img_data,
                                   dx=20):

    lookup = {}
    inactive_mask = get_inactive_mask(img_data.shape, raw_roi_list)
    for roi in raw_roi_list:
        mu, sigma = get_inactive_dist(img_data, roi, inactive_mask, dx=dx)
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
            msg = f'{roi_id} duplicated in '
            msg += 'segmentation_roi_lookup'
            raise RuntimeError(msg)

        lookup[new_roi.roi_id] = new_roi

    return lookup


def find_neighbors(seed_roi,
                   neighbor_lookup,
                   have_been_merged):

    neighbors = []
    for n in neighbor_lookup[seed_roi.roi_id]:
        if n not in have_been_merged:
            neighbors.append(n)
    for ancestor in seed_roi.ancestors:
        for n in neighbor_lookup[ancestor.roi_id]:
            if n not in have_been_merged:
                neighbors.append(n)

    return neighbors


def _get_rings(roi):
    """
    Returns list of lists.
    Each sub list is a ring around the "mountain"
    Contains tuples of the form (last_step, this_step)
    """
    eps = 0.001

    peak = roi.peak
    rings = [[(None, peak.roi_id)]]
    have_seen = set()
    have_seen.add(peak.roi_id)
    while len(have_seen) < len(roi.ancestors):
        last_ring = rings[-1]
        this_ring = []
        for a in roi.ancestors:
            if a.roi_id in have_seen:
                continue

            keep_it = False
            prev = None
            for r_pair in last_ring:
                r = roi.get_ancestor(r_pair[1])
                if do_rois_abut(r, a, dpix=np.sqrt(2)):
                    if r.flux_value >= (a.flux_value+eps):
                        prev = r.roi_id
                        keep_it = True
                        break
            if keep_it:
                this_ring.append((prev, a.roi_id))
                have_seen.add(a.roi_id)

        if len(this_ring) == 0:
            msg = f"empty ring {len(have_seen)} "
            msg += f"{len(roi.ancestors)}"

            for a in roi.ancestors:
                if a.roi_id in have_seen:
                    continue
                msg += f'\n{roi.roi_id} -- ancestor {a.roi_id} {a.x0} {a.y0}'
                for pair in last_ring:
                    up = roi.get_ancestor(pair[1])
                    abut = do_rois_abut(up, a, dpix=np.sqrt(2))
                    msg += f'\n    {up.roi_id} -- {abut} -- '
                    msg += f'{up.flux_value} -- {a.flux_value}'
            raise RuntimeError(msg)

        rings.append(this_ring)
    return rings


def validate_merger(seed_roi, child):
    eps = 0.001

    if not do_rois_abut(seed_roi, child):
        return None

    rings = _get_rings(seed_roi)

    intersection = []
    for ii in range(len(rings)-1,-1,-1):
        this_ring = rings[ii]
        for pair0 in this_ring:
            id0 = pair0[1]
            r0 = seed_roi.get_ancestor(id0)
            if do_rois_abut(r0, child, dpix=np.sqrt(2)):
                if r0.flux_value >= (child.flux_value+eps):
                    return True
    return False


def do_geometric_merger(raw_roi_list,
                        img_data,
                        n_processors,
                        diagnostic_dir = None):

    t0 = time.time()
    merger_candidates = find_merger_candidates(raw_roi_list,
                                               np.sqrt(2.0),
                                               rois_to_ignore=None,
                                               n_processors=n_processors)

    # create a look up table mapping from roi_id to all of the
    # ROI's neighbors
    neighbor_lookup = {}
    for pair in merger_candidates:
        roi0 = pair[0]
        roi1 = pair[1]
        if roi0 not in neighbor_lookup:
            neighbor_lookup[roi0] = set()
        if roi1 not in neighbor_lookup:
            neighbor_lookup[roi1] = set()
        neighbor_lookup[roi0].add(roi1)
        neighbor_lookup[roi1].add(roi0)

    logger.info(f'found {len(merger_candidates)} merger_candidates'
                f' in {time.time()-t0:.2f} seconds')

    # create a lookup table of SegmentationROIs
    t0 = time.time()
    roi_lookup = create_segmentation_roi_lookup(raw_roi_list,
                                                img_data,
                                                dx=20)
    logger.info(f'created roi lookup in {time.time()-t0:.2f} seconds')

    # find all ROIs that are brighter than their neighbors
    t0 = time.time()
    seed_list = []
    for candidate in neighbor_lookup:
        is_seed = True
        f0 = roi_lookup[candidate].flux_value
        for neighbor in neighbor_lookup[candidate]:
            if roi_lookup[neighbor].flux_value > f0:
                is_seed = False
                break
        if is_seed:
            seed_list.append(candidate)

    logger.info(f'got {len(seed_list)} seeds in {time.time()-t0:2f} seconds')

    if diagnostic_dir is not None:
        seed_file = diagnostic_dir / f'seeds.json'
        seed_rois = [ophys_roi_to_extract_roi(roi_lookup[cc])
                     for cc in seed_list]
        with open(seed_file, 'w') as out_file:
            out_file.write(json.dumps(seed_rois, indent=2))

    logger.info(f'plotted {len(seed_list)} seeds in {time.time()-t0:2f} seconds')

    t0 = time.time()
    logger.info('starting merger')
    keep_going = True
    have_been_merged = set()
    i_pass = -1
    incoming_rois = list(roi_lookup.keys())

    t0 = time.time()
    _children = {}
    while keep_going and len(seed_list)>0:

        for s in seed_list:
            assert s in roi_lookup

        n0 = len(roi_lookup)
        i_pass += 1

        # keep track of ROIs that have been merged
        merged_pairs = []
        rejected_pairs = []

        # mapping from a potential child ROI to its seed
        child_to_seed = {}
        for seed_id in seed_list:
            neighbors = find_neighbors(roi_lookup[seed_id],
                                       neighbor_lookup,
                                       have_been_merged)

            for n in neighbors:
                if n in seed_list:
                    continue
                if n not in child_to_seed:
                    child_to_seed[n] = []
                child_to_seed[n].append(seed_id)

        # loop over children; find the brightest original seed;
        # that is where you will merge it (as long as the merger
        # does not require you to go "uphill")
        keep_going = False

        for child_id in child_to_seed:
            child_roi = roi_lookup[child_id]
            best_seed = None
            best_seed_flux = None

            for seed_id in child_to_seed[child_id]:

                # if a possible seed got merged, we should
                # move on, considering this merger later
                if seed_id not in roi_lookup:
                    assert keep_going  # should mean a merger already happend
                    best_seed = None
                    break

                seed_roi = roi_lookup[seed_id]
                if not validate_merger(seed_roi, child_roi):
                    rejected_pairs.append((seed_roi, child_roi))
                    continue
                if best_seed is None or seed_roi.flux_value > best_seed_flux:
                    best_seed = seed_id
                    best_seed_flux = seed_roi.flux_value
            if best_seed is None:
                continue
            seed_roi = roi_lookup[best_seed]
            merged_pairs.append((seed_roi, child_roi))
            new_roi = merge_segmentation_rois(seed_roi,
                                              child_roi,
                                              seed_roi.roi_id,
                                              seed_roi.flux_value)
            assert len(new_roi.ancestors) > 0
            _children[child_id] = roi_lookup.pop(child_id)
            roi_lookup[best_seed] = new_roi
            have_been_merged.add(child_id)
            keep_going = True

        for ii in range(len(seed_list)-1,-1,-1):
            if seed_list[ii] not in roi_lookup:
                seed_list.pop(ii)

        logger.info(f'merged {n0} ROIs to {len(roi_lookup)} '
                    f'after {time.time()-t0:.2f} seconds')


    # now just need to validate that all input rois were preserved

    new_roi_list = []
    for roi_id in incoming_rois:
        if roi_id not in have_been_merged:
            new_roi_list.append(roi_lookup[roi_id])
    return new_roi_list
