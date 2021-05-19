import networkx as nx
import numpy as np
import multiprocessing
from scipy.spatial.distance import cdist
import h5py
import pathlib
import time

from sklearn.decomposition import PCA

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
            obj = {'center': pixel,
                   'rows': (rowmin, rowmax),
                   'cols': (colmin, colmax)}
            p = obj['center']
            assert not img_masked.mask[candidate]
            output.append(obj)
            #print('peaks ',len(output),img_masked.mask.sum())
            for irow in range(rowmin, rowmax):
                for icol in range(colmin, colmax):
                    ii = np.ravel_multi_index((irow, icol), shape)
                    img_masked.mask[ii] = True

            assert img_masked.mask[candidate]

    print('peaks ',len(output))
    return output


def correlate_chunk(data,
                    seed_pt,
                    filter_fraction,
                    rng=None,
                    pixel_ignore=None):
    t0 = time.time()
    global_mask = []
    discard = 1.0-filter_fraction

    n_rows = data.shape[1]
    n_cols = data.shape[2]

    trace = data[:, seed_pt[0], seed_pt[1]]
    thresh = np.quantile(trace, discard)
    mask = np.where(trace>=thresh)[0]
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
        #print(len(mask))
        global_mask.append(mask)
    global_mask = np.unique(np.concatenate(global_mask))
    #print('got global mask',len(global_mask), data.shape[0], time.time()-t0)

    data = data[global_mask, :, :]
    shape = data.shape
    n_pixels = shape[1]*shape[2]
    n_time = shape[0]
    traces = data.reshape(n_time, n_pixels).astype(float)
    del data

    if pixel_ignore is not None:
        traces = traces[:, np.logical_not(pixel_ignore.flatten())]
    n_pixels = np.logical_not(pixel_ignore).sum()
    assert traces.shape[1] == n_pixels
    mu = np.mean(traces, axis=0)
    assert mu.shape == (traces.shape[1],)
    traces -= mu
    var = np.mean(traces**2, axis=0)
    assert var.shape == (traces.shape[1],)
    traces = traces.transpose()
    assert traces.shape == (n_pixels, n_time)

    pearson = np.ones((n_pixels,
                       n_pixels),
                      dtype=float)
    for ii in range(n_pixels):
        trace0 = traces[ii, :]
        numerators = np.mean(traces[ii+1:,:]*trace0, axis=1)
        assert numerators.shape == (n_pixels-1-ii,)
        denominators = np.sqrt(var[ii]*var[ii+1:])
        p = numerators/denominators
        pearson[ii,ii+1:] = p
        pearson[ii+1:, ii] = p

        #if ii % 200 == 0:
        #    print('pearson ii %d %e -- of %d' % (ii, time.time()-t0,
        #                                         n_pixels))
    #print('got pearson ',time.time()-t0)

    wgt = np.copy(pearson)
    assert wgt.shape == (n_pixels, n_pixels)

    pearson_mins = np.min(pearson, axis=0)
    assert pearson_mins[45] == pearson[:,45].min()
    for ii in range(n_pixels):
        wgt[:, ii] = wgt[:, ii]-pearson_mins[ii]

    pearson_norms = np.median(wgt, axis=0)
    assert pearson_norms[11] == np.median(wgt[:, 11])
    assert pearson_norms.shape == (n_pixels,)
    assert (pearson_norms>0.0).all()
    #print('norms ',np.unique(pearson_norms))

    for ii in range(n_pixels):
        wgt[:, ii] = wgt[:, ii]/pearson_norms[ii]

    #print('correlate chunk took %e seconds' % (time.time()-t0))
    #print('wgt: ',wgt.min(),np.median(wgt),wgt.max())

    return wgt, pearson

"""
Try defining background as 75th percentile of not-ROI distance from ROI
Add pixel with greatest ratio of bckd_dist/roi_dist, minimum of 2
re-define

keep going until no one makes ratio of 2 (?)


what about limiting time steps (just taken from brightest pixel
and origin?)
"""


class PotentialROI(object):

    def __init__(self,
                 seed_pt,
                 origin,
                 data,
                 filter_fraction,
                 pixel_ignore=None,
                 diagnostic=False,
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

        self.raw_wgt, pearson = correlate_chunk(data,
                                                self.seed_pt,
                                                filter_fraction,
                                                pixel_ignore=pixel_ignore,
                                                rng=rng)

        assert self.raw_wgt.shape == (self.n_pixels, self.n_pixels)

        self.roi_mask = np.zeros(self.n_pixels, dtype=bool)
        self.roi_mask[self.pixel_to_index[self.seed_pt]] = True

        if not diagnostic:
            return None

        i_seed = self.pixel_to_index[self.seed_pt]
        distances = np.zeros(self.n_pixels)
        for i_pixel in range(self.n_pixels):
            d = np.sqrt(np.sum((self.raw_wgt[i_seed,:]-self.raw_wgt[i_pixel,:])**2))
            distances[i_pixel] = d
        print('distances ',distances.min(),np.median(distances),distances.max())
        print(self.raw_wgt.shape)

        self.distances = distances
        self.mask_img = np.zeros(self.img_shape, dtype=float)
        dmax = self.distances.max()
        for i_pixel in range(self.n_pixels):
            #v = 1.0-self.distances[i_pixel]/dmax
            v = self.distances[i_pixel]
            p = self.index_to_pixel[i_pixel]
            self.mask_img[p[0], p[1]] = v

        b_dist = np.zeros(self.n_pixels, dtype=float)
        i_b = self.pixel_to_index[(0,0)]
        for i_pixel in range(self.n_pixels):
            d = np.sqrt(np.sum((self.raw_wgt[i_b,:]-self.raw_wgt[i_pixel,:])**2))
            b_dist[i_pixel] = d
        self.bckgd_img = np.zeros(self.img_shape, dtype=float)
        self.background_distances = b_dist
        dmax = b_dist.max()
        for i_pixel in range(self.n_pixels):
            #v = 1.0-b_dist[i_pixel]/dmax
            v = b_dist[i_pixel]
            p = self.index_to_pixel[i_pixel]
            self.bckgd_img[p[0], p[1]] = v

        cut_on_bckgd = np.where(b_dist>4)
        self.proto_mask = np.zeros(self.img_shape, dtype=float)
        for i_pixel in range(self.n_pixels):
            d_seed = self.distances[i_pixel]
            d_bckgd = b_dist[i_pixel]
            p = self.index_to_pixel[i_pixel]
            if d_seed<3:
                self.proto_mask[p[0],p[1]] = 1.0
            #self.proto_mask[p[0], p[1]] = d_bckgd/(0.25+d_seed)
            #if d_seed<d_bckgd:
            #    self.proto_mask[p[0], p[1]] = 1.0
        self.proto_mask = self.proto_mask/self.proto_mask.max()

        rough_cut = np.where(self.distances<3)[0]
        rough_wgts = self.raw_wgt[rough_cut, :]
        self.rough_img = np.zeros(self.img_shape, dtype=float)
        for i_pixel in range(self.n_pixels):
            p = self.index_to_pixel[i_pixel]
            d = np.sqrt(np.sum((rough_wgts-self.raw_wgt[i_pixel, :])**2, axis=1))
            assert d.shape == (len(rough_cut),)
            d_min = d.min()
            self.rough_img[p[0], p[1]] = d_min

        s = self.raw_wgt[i_seed,:]
        print('seed ',s.min(),np.median(s),s.max(),
              (s>0.5).sum())

        s = self.raw_wgt[i_b,:]

        print('bckgd ',s.min(),np.median(s),s.max(),
              (s>0.5).sum())

        print('rough img ',self.rough_img.min(),
              np.median(self.rough_img),
              self.rough_img.max())

    def get_not_roi_mask(self):
        roi = self.raw_wgt[self.roi_mask, :]
        complement = np.logical_not(self.roi_mask)
        complement_dexes = np.arange(self.n_pixels, dtype=int)[complement]
        n_complement = complement.sum()
        n_roi = roi.shape[0]
        complement_distances = np.zeros(n_complement, dtype=float)
        i_complement = 0
        for i_complement, i_pixel in enumerate(np.arange(self.n_pixels, dtype=int)[complement]):
            complement_distances[i_complement] = self.get_roi_dist(i_pixel)
        t45 = np.quantile(complement_distances, 0.45)
        t55 = np.quantile(complement_distances, 0.55)
        valid = np.logical_and(complement_distances>t45, complement_distances<t55)
        valid_dexes = complement_dexes[valid]
        self.not_roi_mask = np.zeros(self.n_pixels, dtype=bool)
        self.not_roi_mask[valid_dexes] = True
        #print('not roi ',self.not_roi_mask.sum())

    def _get_masked_dist(self, i_pixel, mask):
        d = np.sqrt(np.sum((self.raw_wgt[mask,:]
                             -self.raw_wgt[i_pixel, :])**2, axis=1))
        assert d.shape == (mask.sum(),)
        return d.min()


    def get_roi_dist(self, i_pixel):
        return self._get_masked_dist(i_pixel, self.roi_mask)

    def get_not_roi_dist(self, i_pixel):
        return self._get_masked_dist(i_pixel, self.not_roi_mask)

    def select_pixels(self) -> bool:
        chose_one = False
        self.get_not_roi_mask()
        d_roi = np.zeros(self.n_pixels, dtype=float)
        d_bckgd = np.zeros(self.n_pixels, dtype=float)
        for i_pixel in range(self.n_pixels):
            if self.roi_mask[i_pixel]:
                d_bckgd[i_pixel] = 0.0
                d_roi[i_pixel] = 999.0
                continue
            d_bckgd[i_pixel] = self.get_not_roi_dist(i_pixel)
            d_roi[i_pixel] = self.get_roi_dist(i_pixel)

        valid = (d_bckgd > 10.0*d_roi)
        if valid.sum() > 0:
            chose_one = True
            self.roi_mask[valid] = True

        return chose_one

    def get_mask(self):
        keep_going = True
        while keep_going:
            #n_roi_0 = self.roi_mask.sum()
            keep_going = self.select_pixels()
            #n_roi_1 = self.roi_mask.sum()
            #print('roi ',n_roi_1,n_roi_1-n_roi_0)

        output_img = np.zeros(self.img_shape, dtype=int)
        for i_pixel in range(self.n_pixels):
            if not self.roi_mask[i_pixel]:
                continue
            p = self.index_to_pixel[i_pixel]
            output_img[p[0], p[1]] = 1
        return output_img


def _get_roi(seed_obj,
             video_path,
             filter_fraction,
             pixel_ignore,
             output_dict,
             roi_id,
             lock):
    t0 = time.time()
    seed_pt = seed_obj['center']
    origin = (seed_obj['rows'][0], seed_obj['cols'][0])
    r = seed_obj['rows']
    c = seed_obj['cols']
    with lock as context:
        t1 = time.time()
        with h5py.File(video_path, 'r') as in_file:
            video_data = in_file['data'][:,
                                         r[0]:r[1],
                                         c[0]:c[1]]
        print('%d read data in %e' % (roi_id, time.time()-t1))


    roi = PotentialROI(seed_pt,
                       origin,
                       video_data,
                       filter_fraction,
                       pixel_ignore=pixel_ignore,
                       diagnostic=False)

    final_mask = roi.get_mask()

    output_dict[roi_id] = (origin, final_mask)
    duration = time.time()-t0
    print('finished %d with %d pixels in %e seconds' %
    (roi_id,final_mask.sum(),duration))


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

    def _run(self, img_data):

        seed_list = find_peaks(img_data,
                               mask=self.roi_pixels,
                               slop=20)

        p_list = []
        mgr = multiprocessing.Manager()
        mgr_dict = mgr.dict()
        mgr_lock = mgr.Lock()
        print('looping over processes')
        for i_seed, seed in enumerate(seed_list):
            center = seed['center']
            mask = self.roi_pixels[seed['rows'][0]:seed['rows'][1],
                                   seed['cols'][0]:seed['cols'][1]]

            p = multiprocessing.Process(target=_get_roi,
                                        args=(seed,
                                              self._video_path,
                                              self._filter_fraction,
                                              mask,
                                              mgr_dict,
                                              i_seed,
                                              mgr_lock))
            p.start()
            p_list.append(p)
            while len(p_list) >= self.n_processors-1:
                to_pop = []
                for ii in range(len(p_list)-1,-1,-1):
                    if p_list[ii].exitcode is not None:
                        to_pop.append(ii)
                for ii in to_pop:
                    p_list.pop(ii)
        for p in p_list:
            p.join()

        for roi_id in mgr_dict:
            origin = mgr_dict[roi_id][0]
            mask = mgr_dict[roi_id][1]
            for ir in range(mask.shape[0]):
                rr = origin[0]+ir
                for ic in range(mask.shape[1]):
                    cc = origin[1]+ic
                    if mask[ir,ic]:
                        self.roi_pixels[rr, cc] = True

    def run(self):
        img_data = graph_to_img(self._graph_path,
                                attribute=self._attribute)
        self.roi_pixels = np.zeros(img_data.shape, dtype=bool)
        keep_going = True
        i_pass = 0
        while keep_going:
            n_roi_0 = self.roi_pixels.sum()
            self._run(img_data)
            n_roi_1 = self.roi_pixels.sum()
            np.savez(f'roi_pixels_{i_pass}.npz', roi=self.roi_pixels)
            i_pass += 1
            if n_roi_1 <= n_roi_0:
                keep_going = False
