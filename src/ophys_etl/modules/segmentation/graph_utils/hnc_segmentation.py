import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
import multiprocessing
from scipy.spatial.distance import cdist
import h5py
import pathlib
import time
import json

from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

import logging

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def graph_to_img(graph_path: pathlib.Path,
                 attribute: str = 'filtered_hnc_Gaussian') -> np.ndarray:
    """
    Convert a graph into a np.ndarray image
    """
    graph = nx.read_gpickle(graph_path)
    node_coords = np.array(graph.nodes).T
    row_max = node_coords[0].max()
    col_max = node_coords[1].max()
    img = np.zeros((row_max+1, col_max+1), dtype=float)
    for node in graph.nodes:
        vals = [graph[node][i][attribute] for i in graph.neighbors(node)]
        img[node[0], node[1]] = np.sum(vals)
    return img


def find_a_peak(img_masked, mu, sigma):
    candidate = np.argmax(img_masked)
    if img_masked[candidate] > mu+2*sigma:
        return candidate
    return None


def find_peaks(img, mask=None, slop=20):
    output = []
    shape = img.shape
    img_flat = img.flatten()

    if mask is None:
        mask_flat = np.zeros(img_flat.shape, dtype=bool)
    else:
        mask_flat = mask.flatten()

    img_masked = np.ma.masked_array(img_flat, mask=mask_flat)
    i25 = np.quantile(img_masked, 0.25)
    i75 = np.quantile(img_masked, 0.75)
    sigma = (i75-i25)/1.349
    mu = np.median(img_masked)

    keep_going = True
    while keep_going:
        candidate = find_a_peak(img_masked, mu, sigma)
        if candidate is None:
            keep_going = False
        else:
            pixel = np.unravel_index(candidate, shape)
            rowmin = max(0, pixel[0]-slop)
            rowmax = min(shape[0], pixel[0]+slop)
            colmin = max(0, pixel[1]-slop)
            colmax = min(shape[1], pixel[1]+slop)

            obj = {'center': (int(pixel[0]), int(pixel[1])),
                   'rows': (int(rowmin), int(rowmax)),
                   'cols': (int(colmin), int(colmax))}
            output.append(obj)
            for irow in range(rowmin, rowmax):
                for icol in range(colmin, colmax):
                    ii = np.ravel_multi_index((irow, icol), shape)
                    img_masked.mask[ii] = True

    return output


def correlate_chunk(data,
                    seed_pt,
                    filter_fraction,
                    rng=None,
                    pixel_ignore=None):
    global_mask = []
    discard = 1.0-filter_fraction

    n_rows = data.shape[1]
    n_cols = data.shape[2]

    trace = data[:, seed_pt[0], seed_pt[1]]
    thresh = np.quantile(trace, discard)
    mask = np.where(trace >= thresh)[0]
    global_mask.append(mask)

    n_seeds = 10
    if rng is None:
        rng = np.random.RandomState(87123)
    chosen = []
    while len(chosen) < n_seeds:
        c = rng.choice(np.arange(n_rows*n_cols, dtype=int),
                       size=1, replace=False)
        p = np.unravel_index(c, data.shape[1:])
        if pixel_ignore is None or not pixel_ignore[p[0], p[1]]:
            chosen.append(p)

    for chosen_pixel in chosen:
        trace = data[:, chosen_pixel[0], chosen_pixel[1]]
        thresh = np.quantile(trace, discard)
        mask = np.where(trace >= thresh)[0]
        global_mask.append(mask)
    global_mask = np.unique(np.concatenate(global_mask))

    data = data[global_mask, :, :]
    shape = data.shape
    n_pixels = shape[1]*shape[2]
    n_time = shape[0]
    traces = data.reshape(n_time, n_pixels).astype(float)
    del data

    if pixel_ignore is not None:
        traces = traces[:, np.logical_not(pixel_ignore.flatten())]
    n_pixels = np.logical_not(pixel_ignore).sum()
    mu = np.mean(traces, axis=0)
    traces -= mu
    var = np.mean(traces**2, axis=0)
    traces = traces.transpose()

    pearson = np.ones((n_pixels,
                       n_pixels),
                      dtype=float)

    numerators = np.tensordot(traces, traces, axes=(1, 1))/n_time

    for ii in range(n_pixels):
        local_numerators = numerators[ii, ii+1:]
        denominators = np.sqrt(var[ii]*var[ii+1:])
        p = local_numerators/denominators
        pearson[ii, ii+1:] = p
        pearson[ii+1:, ii] = p

    wgt = np.copy(pearson)

    pearson_mins = np.min(pearson, axis=0)
    for ii in range(n_pixels):
        wgt[:, ii] = wgt[:, ii]-pearson_mins[ii]

    p75 = np.quantile(wgt, 0.75, axis=0)
    p25 = np.quantile(wgt, 0.25, axis=0)
    pmax = np.max(wgt, axis=0)
    pmin = np.min(wgt, axis=0)

    pearson_norms = p75-p25
    pearson_norms = np.where(pearson_norms>1.0e-20,
                             pearson_norms,
                             pmax-pmin)

    pearson_norms = np.where(pearson_norms>1.0e-20,
                             pearson_norms,
                             1.0)

    for ii in range(n_pixels):
        wgt[:, ii] = wgt[:, ii]/pearson_norms[ii]

    distances = cdist(wgt, wgt, metric='euclidean')
    return distances, pearson


class PotentialROI(object):

    def __init__(self,
                 seed_pt,
                 origin,
                 data,
                 filter_fraction,
                 pixel_ignore=None,
                 rng=None):
        """
        seed and origin are in full-plane coordinates
        """

        self.origin = origin
        self.seed_pt = (seed_pt[0]-origin[0], seed_pt[1]-origin[1])
        self.filter_fraction = filter_fraction
        self.img_shape = data.shape[1:]

        self.index_to_pixel = []
        self.pixel_to_index = {}
        if pixel_ignore is not None:
            pixel_ignore_flat = pixel_ignore.flatten()

        self.n_pixels = 0
        for ii in range(self.img_shape[0]*self.img_shape[1]):
            if pixel_ignore is None or not pixel_ignore_flat[ii]:
                p = np.unravel_index(ii, self.img_shape)
                self.index_to_pixel.append(p)
                self.pixel_to_index[p] = len(self.index_to_pixel)-1
                self.n_pixels += 1

        (self.feature_distances,
         pearson) = correlate_chunk(data,
                                    self.seed_pt,
                                    filter_fraction,
                                    pixel_ignore=pixel_ignore,
                                    rng=rng)

        self.roi_mask = np.zeros(self.n_pixels, dtype=bool)
        self.roi_mask[self.pixel_to_index[self.seed_pt]] = True

    def get_not_roi_mask(self):
        complement = np.logical_not(self.roi_mask)
        n_complement = complement.sum()
        complement_dexes = np.arange(self.n_pixels, dtype=int)[complement]

        complement_distances = self.feature_distances[complement, :]
        complement_distances = complement_distances[:, self.roi_mask]
        if len(complement_distances.shape) > 1:
            complement_distances = complement_distances.min(axis=1)

        self.not_roi_mask = np.zeros(self.n_pixels, dtype=bool)
        if n_complement < 10:
            self.not_roi_mask[complement] = True
        else:
            t10 = np.quantile(complement_distances, 0.1)
            valid = complement_distances > t10
            valid_dexes = complement_dexes[valid]
            self.not_roi_mask[valid_dexes] = True

    def select_pixels(self) -> bool:
        chose_one = False
        self.get_not_roi_mask()

        d_roi = np.median(self.feature_distances[:, self.roi_mask], axis=1)
        d_roi[self.roi_mask] = 999.0

        # take the mean of as many background points as there are
        # ROI points, in case one set is dominated by outliers
        d_bckgd = np.sort(self.feature_distances[:, self.not_roi_mask], axis=1)
        n_roi = self.roi_mask.sum()
        d_bckgd = np.median(d_bckgd[:, :n_roi], axis=1)
        d_bckgd[self.roi_mask] = 0.0

        valid = d_bckgd > 2*d_roi
        if valid.sum() > 0:
            chose_one = True
            self.roi_mask[valid] = True

        return chose_one

    def get_mask(self):
        keep_going = True
        while keep_going:
            keep_going = self.select_pixels()

        output_img = np.zeros(self.img_shape, dtype=int)
        for i_pixel in range(self.n_pixels):
            if not self.roi_mask[i_pixel]:
                continue
            p = self.index_to_pixel[i_pixel]
            output_img[p[0], p[1]] = 1

        return output_img


def _get_roi(seed_obj,
             video_data,
             filter_fraction,
             pixel_ignore,
             output_dict,
             roi_id):
    seed_pt = seed_obj['center']
    origin = (seed_obj['rows'][0], seed_obj['cols'][0])

    roi = PotentialROI(seed_pt,
                       origin,
                       video_data,
                       filter_fraction,
                       pixel_ignore=pixel_ignore)

    final_mask = roi.get_mask()
    output_dict[roi_id] = (origin, final_mask)


def convert_to_lims_roi(origin, mask, roi_id=0):

    # trim mask
    row0 = 0
    col0 = 0
    row1 = mask.shape[0]
    col1 = mask.shape[1]
    for i_row in range(mask.shape[0]):
        if mask[i_row, :].sum() > 0:
            break
        row0 += 1
    for i_row in range(mask.shape[0]-1, -1, -1):
        if mask[i_row, :].sum() > 0:
            break
        row1 -= 1
    for i_col in range(mask.shape[1]):
        if mask[:, i_col].sum() > 0:
            break
        col0 += 1
    for i_col in range(mask.shape[1]-1, -1, -1):
        if mask[:, i_col].sum() > 0:
            break
        col1 -= 1

    new_mask = mask[row0:row1, col0:col1]
    roi = ExtractROI(id=roi_id,
                     x=int(origin[1]+col0),
                     y=int(origin[0]+row0),
                     width=int(col1-col0),
                     height=int(row1-row0),
                     valid=False,
                     mask=[i.tolist() for i in new_mask])
    return roi


def create_roi_plot(plot_path, img_data, roi_list):
    fig, axes = plt.subplots(1, 2, figsize=(40, 20))
    axes[0].imshow(img_data)
    axes[1].imshow(img_data)

    bdry_pixels = np.zeros(img_data.shape, dtype=int)
    for roi in roi_list:
        ophys_roi = OphysROI(
                        roi_id=0,
                        x0=roi['x'],
                        y0=roi['y'],
                        width=roi['width'],
                        height=roi['height'],
                        valid_roi=False,
                        mask_matrix=roi['mask'])

        bdry = ophys_roi.boundary_mask
        for ir in range(ophys_roi.height):
            for ic in range(ophys_roi.width):
                if bdry[ir, ic]:
                    bdry_pixels[ir+ophys_roi.y0,
                                ic+ophys_roi.x0] = 1

    bdry_pixels = np.ma.masked_where(bdry_pixels == 0,
                                     bdry_pixels)
    axes[1].imshow(bdry_pixels, cmap='autumn', alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig=fig)


class HNCSegmenter(object):

    def __init__(self,
                 graph_path: pathlib.Path,
                 video_path: pathlib.Path,
                 attribute: str = 'filtered_hnc_Gaussian',
                 filter_fraction: float = 0.2,
                 n_processors=8):
        self.n_processors = n_processors
        self._attribute = attribute
        self._graph_path = graph_path
        self._video_path = video_path
        self._filter_fraction = filter_fraction
        self.rng = np.random.RandomState(11923141)
        self._graph_img = graph_to_img(graph_path,
                                       attribute=attribute)

        with h5py.File(self._video_path, 'r') as in_file:
            self._movie_data = in_file['data'][()]
            movie_shape = self._movie_data.shape
            if movie_shape[1:] != self._graph_img.shape:
                msg = f'movie shape: {movie_shape}\n'
                msg += f'img shape: {self._graph_img.shape}'
                raise RuntimeError(msg)

    def _run(self, img_data, video_data):

        seed_list = find_peaks(img_data,
                               mask=self.roi_pixels,
                               slop=20)

        p_list = []
        mgr = multiprocessing.Manager()
        mgr_dict = mgr.dict()
        for i_seed, seed in enumerate(seed_list):
            self.roi_id += 1
            mask = self.roi_pixels[seed['rows'][0]:seed['rows'][1],
                                   seed['cols'][0]:seed['cols'][1]]

            video_data_subset = video_data[:,
                                           seed['rows'][0]:seed['rows'][1],
                                           seed['cols'][0]:seed['cols'][1]]

            p = multiprocessing.Process(target=_get_roi,
                                        args=(seed,
                                              video_data_subset,
                                              self._filter_fraction,
                                              mask,
                                              mgr_dict,
                                              self.roi_id))
            p.start()
            p_list.append(p)
            while len(p_list) >= self.n_processors-1:
                to_pop = []
                for ii in range(len(p_list)-1, -1, -1):
                    if p_list[ii].exitcode is not None:
                        to_pop.append(ii)
                for ii in to_pop:
                    p_list.pop(ii)
        for p in p_list:
            p.join()

        for roi_id in mgr_dict:
            origin = mgr_dict[roi_id][0]
            mask = mgr_dict[roi_id][1]
            roi = convert_to_lims_roi(origin,
                                      mask,
                                      roi_id=roi_id)
            self.roi_list.append(roi)
            for ir in range(mask.shape[0]):
                rr = origin[0]+ir
                for ic in range(mask.shape[1]):
                    cc = origin[1]+ic
                    if mask[ir, ic]:
                        self.roi_pixels[rr, cc] = True
        return seed_list

    def run(self,
            roi_path=None,
            seed_path=None,
            plot_path=None):
        t0 = time.time()

        img_data = graph_to_img(self._graph_path,
                                attribute=self._attribute)

        logger.info(f'read in image data from {str(self._graph_path)}')

        with h5py.File(self._video_path, 'r') as in_file:
            video_data = in_file['data'][()]
        logger.info(f'read in video data from {str(self._video_path)}')

        if seed_path is not None:
            seed_record = {}

        self.roi_list = []
        self.roi_id = -1
        self.roi_pixels = np.zeros(img_data.shape, dtype=bool)
        keep_going = True
        i_pass = 0
        while keep_going:
            n_roi_0 = self.roi_pixels.sum()
            roi_seeds = self._run(img_data, video_data)
            n_roi_1 = self.roi_pixels.sum()
            duration = time.time()-t0
            msg = f'Completed pass with {len(roi_seeds)} ROIs '
            msg += f'in {duration:.2f} seconds; '
            msg += f'{n_roi_1} total ROI pixels'
            logger.info(msg)

            if seed_path is not None:
                seed_record[i_pass] = roi_seeds

            i_pass += 1
            if n_roi_1 <= n_roi_0:
                keep_going = False


        if seed_path is not None:
            with open(seed_path, 'w') as out_file:
                out_file.write(json.dumps(seed_record, indent=2))

        with open(roi_path, 'w') as out_file:
            out_file.write(json.dumps(self.roi_list, indent=2))

        if plot_path is not None:
            create_roi_plot(plot_path, img_data, self.roi_list)

        duration = time.time()-t0
        logger.info(f'Completed segmentation in {duration:.2f} seconds')
