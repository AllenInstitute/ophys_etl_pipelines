import matplotlib.figure as mplt_fig

import numpy as np
import copy
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.postprocess_utils.roi_merging import (
    merge_rois,
    find_merger_candidates,
    do_rois_abut)

import logging
import time

"""
need carry median value in the ROI

if it inherits from OphysROI, we can merge it with pre-existing code
"""


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def _plot_mergers(img_arr: np.ndarray,
                  merger_pairs,
                  out_name):

    n = np.ceil(np.sqrt(len(merger_pairs))).astype(int)
    fig = mplt_fig.Figure(figsize=(n*7, n*7))
    axes = [fig.add_subplot(n,n,i) for i in range(1,len(merger_pairs)+1,1)]
    mx = img_arr.max()
    rgb_img = np.zeros((img_arr.shape[0], img_arr.shape[1], 3),
                       dtype=np.uint8)
    img = np.round(255*img_arr.astype(float)/max(1,mx)).astype(np.uint8)
    for ic in range(3):
        rgb_img[:,:,ic] = img
    del img

    alpha=0.5
    for ii in range(len(merger_pairs)):
        roi0 = merger_pairs[ii][0]
        roi1 = merger_pairs[ii][1]
        img = np.copy(rgb_img)

        npix = roi0.mask_matrix.sum()+roi1.mask_matrix.sum()

        for roi, color in zip((roi0, roi1),[(255,0,0),(0,255,0)]):
            msk = roi.mask_matrix
            for ir in range(roi.height):
                for ic in range(roi.width):
                    if not msk[ir, ic]:
                        continue
                    row = ir+roi.y0
                    col = ic+roi.x0
                    for jj in range(3):
                        old = rgb_img[row, col, jj]
                        new = np.round(alpha*color[jj]+(1.0-alpha)*old)
                        new = np.uint8(new)
                        img[row, col, jj] = new
            axes[ii].imshow(img)

    for jj in range(ii, len(axes), 1):
        axes[jj].tick_params(left=0,bottom=0,labelleft=0,labelbottom=0)
        for s in ('top', 'left', 'bottom','right'):
            axes[jj].spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_name)


def plot_mergers(img_arr: np.ndarray,
                 merger_pairs,
                 out_name):

    n_sub = 16
    for i0 in range(0, len(merger_pairs), n_sub):
        new_out = str(out_name).replace('.png',f'_{i0}.png')
        print(f'{i0} -- {new_out}')
        s_pairs = merger_pairs[i0:i0+n_sub]
        _plot_mergers(img_arr, s_pairs,
                      new_out)



def get_inactive_mask(img_data_shape, roi_list):
    full_mask = np.zeros(img_data_shape, dtype=bool)
    for roi in roi_list:
        mask = roi.mask_matrix
        region = full_mask[roi.y0:roi.y0+roi.height,
                           roi.x0:roi.x0+roi.width]
        full_mask[roi.y0:roi.y0+roi.height,
                  roi.x0:roi.x0+roi.width] = np.logical_or(mask, region)
    return np.logical_not(full_mask)


def get_inactive_dist(img_data, roi, inactive_mask, dx=10):
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

    def __init__(self,
                 x0,
                 y0,
                 height,
                 width,
                 valid_roi,
                 mask_matrix,
                 roi_id,
                 flux_value=0.0,
                 ancestors=None):

        self.flux_value = flux_value

        self.ancestors = []
        if ancestors is not None:
            self.ancestors = []
            for roi in ancestors:
                if len(roi.ancestors) == 0:
                    self.ancestors.append(roi)
                else:
                    for sub_roi in roi.ancestors:
                        assert isinstance(sub_roi, SegmentationROI)
                        self.ancestors.append(sub_roi)

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

    def _create_ancestor_lookup(self):
        lookup = {}
        for a in self.ancestors:
            lookup[a.roi_id] = a
        return lookup

    @classmethod
    def from_ophys_roi(cls, input_roi, ancestors=None, flux_value=0.0):
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
    def peak(self):
        if len(self.ancestors) == 0:
            return self
        peak_val = None
        peak_roi = None
        for roi in self.ancestors:
            if peak_val is None or roi.flux_value > peak_val:
                peak_roi = roi
                peak_val = roi.flux_value
        return peak_roi

    def ancestor_lookup(self, roi_id):
        if roi_id in self._ancestor_lookup:
            return self._ancestor_lookup[roi_id]
        if roi_id != self.roi_id:
            id_list = list(self._ancestor_lookup.keys())
            id_list.append(self.roi_id)
            id_list.sort()
            raise RuntimeError(f"cannot lookup {roi_id}; "
                               f"{id_list}")
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
                r = roi.ancestor_lookup(r_pair[1])
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
                    up = roi.ancestor_lookup(pair[1])
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
            r0 = seed_roi.ancestor_lookup(id0)
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
                                               unchanged_rois=None,
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

    """
    seed_pairs = []
    for ii in range(0, len(seed_list), 2):
        if ii+1 < len(seed_list)-1:
            roi0 = roi_lookup[seed_list[ii+1]]
        else:
            roi0 = roi_lookup[seed_list[ii-1]]
        roi1 = roi_lookup[seed_list[ii]]
        seed_pairs.append((roi0, roi1))
        accepted_file = diagnostic_dir / f'seeds.png'
    plot_mergers(img_data,
                 seed_pairs,
                 accepted_file)
    """

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

        """
        if diagnostic_dir is not None:
            accepted_file = diagnostic_dir / f'accepted_mergers_{i_pass}.png'
            plot_mergers(img_data,
                         merged_pairs,
                         accepted_file)
        """

        logger.info(f'merged {n0} ROIs to {len(roi_lookup)} '
                    f'after {time.time()-t0:.2f} seconds')


    # now just need to validate that all input rois were preserved

    new_roi_list = []
    for roi_id in incoming_rois:
        if roi_id not in have_been_merged:
            new_roi_list.append(roi_lookup[roi_id])
    return new_roi_list
