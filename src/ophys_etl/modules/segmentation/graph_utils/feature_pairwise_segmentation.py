from typing import Tuple, List, Optional
import numpy as np
import multiprocessing
import pathlib
import time
import h5py
import json

from scipy.spatial.distance import cdist

from ophys_etl.modules.segmentation.\
    graph_utils.feature_vector_segmentation import (
        find_peaks,
        FeatureVectorSegmenter,
        convert_to_lims_roi)

from ophys_etl.modules.segmentation.\
    graph_utils.feature_vector_rois import (
        PotentialROI,
        normalize_features)

from ophys_etl.modules.segmentation.\
    graph_utils.plotting import (
        graph_to_img,
        create_roi_plot)

from ophys_etl.modules.segmentation.graph_utils.feature_vector_rois import (
    PearsonFeatureROI)


import logging

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def get_pixel_indexes(center, slop, shape):
    row0 = max(0, center[0]-slop)
    row1 = min(shape[0], center[0]+slop+1)
    col0 = max(0, center[1]-slop)
    col1 = min(shape[1], center[1]+slop+1)

    (img_rows,
     img_cols) = np.meshgrid(np.arange(row0, row1, 1, dtype=int),
                             np.arange(col0, col1, 1, dtype=int),
                             indexing='ij')

    coords = np.array([img_rows.flatten(), img_cols.flatten()])

    return np.ravel_multi_index(coords, shape)


def correlate_pixel(pixel_pt: Tuple[int, int],
                    i_pixel_global: int,
                    filter_fraction: float,
                    video_data: np.ndarray,
                    pixel_indexes: np.ndarray,
                    output_dict: dict):
    """
    pixel_pt is not origin subtracted
    video_data is a subset
    """
    n_pixels = video_data.shape[0]
    assert n_pixels == len(pixel_indexes)

    i_pixel = np.where(pixel_indexes == i_pixel_global)[0][0]

    trace = video_data[i_pixel, :]
    threshold = np.quantile(trace, 1.0-filter_fraction)

    valid_time = (trace >= threshold)
    video_data = video_data[:, valid_time].astype(float)
    trace = video_data[i_pixel, :]
    n_time = video_data.shape[1]

    mu = np.mean(trace)
    video_mu = np.mean(video_data, axis=1)
    assert video_mu.shape == (n_pixels, )

    trace -= mu
    video_data = (video_data.T-video_mu).T

    trace_var = np.mean(trace**2)
    video_var = np.mean(video_data**2, axis=1)

    num = np.dot(video_data, trace)/n_time
    denom = np.sqrt(trace_var*video_var)
    corr = num/denom
    output_dict[pixel_pt] = {'corr': corr,
                             'pixels': pixel_indexes}


def correlate_tile(video_path: pathlib.Path,
                   rowbounds: Tuple[int, int],
                   colbounds: Tuple[int, int],
                   img_shape: Tuple[int, int],
                   slop: int,
                   filter_fraction: float,
                   output_dict: multiprocessing.managers.DictProxy,
                   done_dict: multiprocessing.managers.DictProxy,
                   lock):

    local_output_dict = {}

    slop = slop*2

    rowmin = max(0, rowbounds[0]-slop)
    rowmax = min(img_shape[0], rowbounds[1]+slop+1)
    colmin = max(0, colbounds[0]-slop)
    colmax = min(img_shape[1], colbounds[1]+slop+1)
    with h5py.File(video_path, 'r') as in_file:
        video_data = in_file['data'][:, rowmin:rowmax, colmin:colmax]

    (img_rows,
     img_cols) = np.meshgrid(np.arange(rowmin, rowmax, 1, dtype=int),
                             np.arange(colmin, colmax, 1, dtype=int),
                             indexing='ij')

    img_rows = img_rows.flatten()
    img_cols = img_cols.flatten()
    pixel_indexes = np.ravel_multi_index(np.array([img_rows, img_cols]),
                                         img_shape)

    video_data = video_data.reshape(video_data.shape[0], -1)
    video_data = video_data.transpose()

    n_tot = (rowbounds[1]-rowbounds[0])*(colbounds[1]-colbounds[0])
    t0 = time.time()
    for row in range(rowbounds[0], rowbounds[1], 1):
        for col in range(colbounds[0], colbounds[1], 1):
            pixel = (row, col)
            i_pixel = np.ravel_multi_index(pixel, img_shape)

            r0 = max(0, row-slop)
            r1 = min(img_shape[0], row+slop+1)
            c0 = max(0, col-slop)
            c1 = min(img_shape[1], col+slop+1)

            (img_rows,
             img_cols) = np.meshgrid(np.arange(r0-rowmin, r1-rowmin,
                                               1, dtype=int),
                                     np.arange(c0-colmin, c1-colmin,
                                               1, dtype=int),
                                     indexing='ij')

            local_indexes = np.ravel_multi_index([img_rows.flatten(),
                                                  img_cols.flatten()],
                                                 (rowmax-rowmin,
                                                  colmax-colmin))

            chosen_pixels = pixel_indexes[local_indexes]
            sub_video = video_data[local_indexes, :]

            args = (pixel,
                    i_pixel,
                    filter_fraction,
                    sub_video,
                    chosen_pixels,
                    local_output_dict)

            correlate_pixel(*args)

    # copy results over to output_dict
    key_list = list(local_output_dict.keys())
    with lock:
        for key in key_list:
            obj = local_output_dict.pop(key)
            output_dict[key] = obj
            done_dict[key] = True


class FeaturePairwiseROI(PotentialROI):

    def __init__(self,
                 seed_pt: Tuple[int, int],
                 window_indexes: np.ndarray,
                 corr_indexes: np.ndarray,
                 corr_values: np.ndarray,
                 pixel_ignore: np.ndarray):

        self.img_shape = pixel_ignore.shape
        pixel_ignore = pixel_ignore.flatten()
        self.pixel_threshold = 1.5

        valid_pixel_indexes = np.copy(window_indexes)
        valid_pixel_mask = np.logical_not(pixel_ignore[valid_pixel_indexes])
        valid_pixel_indexes = valid_pixel_indexes[valid_pixel_mask]
        self.n_pixels = valid_pixel_mask.sum()

        self.index_to_pixel = []
        self.pixel_to_index = {}
        for ii, index, in enumerate(valid_pixel_indexes):
            row, col = np.unravel_index(index, self.img_shape)
            self.index_to_pixel.append((row, col))
            self.pixel_to_index[(row, col)] = ii

        self.roi_mask = np.zeros(self.n_pixels, dtype=bool)
        self.roi_mask[self.pixel_to_index[seed_pt]] = True

        features = np.zeros((self.n_pixels, self.n_pixels), dtype=float)
        for ii, ipx in enumerate(valid_pixel_indexes):
            jj = np.where(window_indexes == ipx)[0][0]
            fv_pix = corr_indexes[jj, :]
            fv_vals = corr_values[jj, :]
            mask = np.where(fv_pix >= 0)
            fv_pix = fv_pix[mask]
            fv_vals = fv_vals[mask]
            chosen = np.searchsorted(fv_pix, valid_pixel_indexes)
            np.testing.assert_array_equal(fv_pix[chosen], valid_pixel_indexes)
            features[ii, :] = fv_vals[chosen]

        features = normalize_features(features)
        self.feature_distances = cdist(features,
                                       features,
                                       metric='euclidean')


def transcribe_data(correlated_pixels,
                    correlated_values,
                    results_dict,
                    done_dict,
                    img_shape,
                    lock):
    with lock:
        n_transcribed = 0
        key_list = list(done_dict.keys())
        for key in key_list:
            if not done_dict[key]:
                continue
            done_dict.pop(key)
            obj = results_dict.pop(key)
            n = len(obj['pixels'])
            if n >= correlated_pixels.shape[1]:
                raise RuntimeError(f'pixel {ii} has {n} neighors; '
                                   'destination shape '
                                   f'{correlated_pixels.shape}')
            ii = np.ravel_multi_index(key, img_shape)
            if ii>= correlated_pixels.shape[0]:
                raise RuntimeError(f'transcribing pixel {ii} '
                                   f'but shape {correlate_pixels.shape}')
            correlated_pixels[ii, :n] = obj['pixels']
            correlated_values[ii, :n] = obj['corr']
            n_transcribed += 1
    return n_transcribed


def _get_roi(roi_id,
             seed,
             window_indexes,
             corr_indexes,
             corr_values,
             roi_pixels,
             roi_list,
             local_masks):
    roi = FeaturePairwiseROI(seed['center'],
                             window_indexes,
                             corr_indexes,
                             corr_values,
                             roi_pixels)

    mask = roi.get_mask()
    lims_roi = convert_to_lims_roi((0, 0), mask, roi_id=roi_id)
    roi_list.append(lims_roi)
    local_masks.append(mask)


class FeaturePairwiseSegmenter(FeatureVectorSegmenter):
    """
    Version of FeatureVectorSegmenter that uses a unique set of timestamps
    to calculate correlation between each pair of pixels
    """
    def __init__(self,
                 graph_input: pathlib.Path,
                 video_input: pathlib.Path,
                 attribute: str = 'filtered_hnc_Gaussian',
                 filter_fraction: float = 0.2,
                 n_processors=8,
                 roi_class=PearsonFeatureROI):

        self.slop = 20

        super().__init__(graph_input,
                         video_input,
                         attribute=attribute,
                         filter_fraction=filter_fraction,
                         n_processors=n_processors,
                         roi_class=roi_class)

    def _load_video(self, video_path: pathlib.Path):
        return np.zeros(3, dtype=float)
        with h5py.File(video_path, 'r') as in_file:
            video_data = in_file['data'][()]
        video_data = video_data.reshape(video_data.shape[0], -1)
        video_data = video_data.transpose()
        return video_data

    def pre_correlate_pixels(self,
                             img_data: np.ndarray) -> List[dict]:
        """
        video data must already be reshaped etc.
        """
        pre_corr_mgr = multiprocessing.Manager()
        pre_corr_lookup = pre_corr_mgr.dict()
        pre_corr_done = pre_corr_mgr.dict()
        lock = pre_corr_mgr.Lock()

        # pre-compute correlations
        img_shape = img_data.shape
        sqrt_proc = max(1, np.ceil(np.sqrt(self.n_processors-1)).astype(int))
        drow = max(1, img_shape[0]//(2*sqrt_proc))
        dcol = max(1, img_shape[1]//(2*sqrt_proc))

        n_pixels = img_shape[0]*img_shape[1]
        max_indices = 1+(4*self.slop+1)**2
        correlated_pixels = -1*np.ones((n_pixels, max_indices), dtype=int)
        correlated_values = np.zeros((n_pixels, max_indices), dtype=float)

        n_transcribed = 0

        t0 = time.time()
        p_list = []
        ct_done = 0
        row_list = list(range(0, img_shape[0], drow))
        col_list = list(range(0, img_shape[1], dcol))
        n_tiles = len(row_list)*len(col_list)

        for row0 in row_list:
            row1 = min(img_shape[0], row0+drow)
            for col0 in col_list:
                col1 = min(img_shape[1], col0+dcol)

                args = (self._video_input,
                        (row0, row1),
                        (col0, col1),
                        img_shape,
                        self.slop,
                        self._filter_fraction,
                        pre_corr_lookup,
                        pre_corr_done,
                        lock)

                p = multiprocessing.Process(target=correlate_tile,
                                            args=args)
                p.start()
                p_list.append(p)
                # make sure that all processors are working at all times,
                # if possible


                while len(p_list) > 0 and len(p_list) >= (self.n_processors-1):
                    to_pop = []
                    new_done = 0
                    for ii in range(len(p_list)-1, -1, -1):
                        if p_list[ii].exitcode is not None:
                            to_pop.append(ii)
                    for ii in to_pop:
                        p_list.pop(ii)
                        ct_done += 1
                        new_done += 1

                    if len(pre_corr_done) > 0:
                        n_transcribed += transcribe_data(
                                             correlated_pixels,
                                             correlated_values,
                                             pre_corr_lookup,
                                             pre_corr_done,
                                             img_shape,
                                             lock)
                    if new_done > 0:
                        duration = time.time()-t0
                        logger.info(f'{ct_done} tiles done out of {n_tiles} '
                                    f'({n_transcribed} pixels transcribed) '
                                    f'in {duration:.2f} sec')

        while len(p_list) > 0:
            to_pop = []
            new_done = 0
            for ii in range(len(p_list)-1, -1, -1):
                if p_list[ii].exitcode is not None:
                    to_pop.append(ii)
            for ii in to_pop:
                p_list.pop(ii)
                ct_done += 1
                new_done += 1

            if len(pre_corr_done) > 0:
                n_transcribed += transcribe_data(
                                     correlated_pixels,
                                     correlated_values,
                                     pre_corr_lookup,
                                     pre_corr_done,
                                     img_shape,
                                     lock)

            if new_done > 0:
                duration = time.time()-t0
                logger.info(f'{ct_done} tiles done out of {n_tiles} '
                            f'({n_transcribed} pixels transcribed) '
                            f'in {duration:.2f} sec')

        if len(pre_corr_done) > 0:
            n_transcribed += transcribe_data(
                                 correlated_pixels,
                                 correlated_values,
                                 pre_corr_lookup,
                                 pre_corr_done,
                                 img_shape,
                                 lock)

            duration = time.time()-t0
            logger.info(f'{ct_done} tiles done out of {n_tiles} '
                        f'({n_transcribed} pixels transcribed) '
                        f'in {duration:.2f} sec')

        if n_transcribed != n_pixels:
            raise RuntimeError(f'transcribed {n_transcribed} '
                               f'of {n_pixels} pixels')

        for ii in range(n_pixels):
            n = (correlated_pixels[ii,:]>=0).sum()
            if n == 0:
                raise RuntimeError(f'pixel {ii} has no correlations')

        duration = time.time()-t0
        logger.info('pre correlation took %e seconds' % duration)

        return(correlated_values, correlated_pixels)

    def run(self,
            roi_output: pathlib.Path,
            seed_output: Optional[pathlib.Path] = None,
            plot_output: Optional[pathlib.Path] = None) -> None:

        t0 = time.time()

        img_data = graph_to_img(self._graph_input,
                                attribute_name=self._attribute)

        img_shape = img_data.shape

        (correlated_values,
         correlated_pixels) = self.pre_correlate_pixels(img_data)

        self.roi_pixels = np.zeros(img_data.shape, dtype=bool)

        if seed_output is not None:
            seed_record = {}

        mgr = multiprocessing.Manager()
        roi_list = mgr.list()

        roi_id = -1
        keep_going = True
        i_pass = 0
        while keep_going:
            n_roi_0 = self.roi_pixels.sum()

            seed_list = find_peaks(img_data,
                                   mask=self.roi_pixels,
                                   slop=20)

            if seed_output is not None:
                seed_record[i_pass] = seed_list
            i_pass += 1

            local_masks = mgr.list()
            p_list = []
            ct_done = 0
            n_seeds = len(seed_list)

            logger.info(f'running on {n_seeds} seeds')
            for seed in seed_list:
                roi_id += 1

                window_indexes = get_pixel_indexes(seed['center'],
                                                   self.slop,
                                                   img_shape)

                corr_indexes = correlated_pixels[window_indexes, :]
                corr_values = correlated_values[window_indexes, :]

                args = (roi_id,
                        seed,
                        window_indexes,
                        corr_indexes,
                        corr_values,
                        self.roi_pixels,
                        roi_list,
                        local_masks)

                p = multiprocessing.Process(target=_get_roi,
                                            args=args)
                p.start()
                p_list.append(p)
                while len(p_list) >= (self.n_processors-1):
                    to_pop = []
                    new_done = 0
                    for ii in range(len(p_list)-1, -1, -1):
                        if p_list[ii].exitcode is not None:
                            to_pop.append(ii)
                    for ii in to_pop:
                        p_list.pop(ii)
                        ct_done += 1
                        new_done += 1

                    if new_done > 0:
                        duration = time.time()-t0
                        logger.info(f'{ct_done} rois of {n_seeds} '
                                    f'after {duration:.2f} sec')

            for p in p_list:
                p.join()

            for mask in local_masks:
                self.roi_pixels[mask] = True
            n_roi_1 = self.roi_pixels.sum()

            if n_roi_1 == n_roi_0:
                keep_going = False

            duration = time.time()-t0
            logger.info(f'found {len(seed_list)} seeds; {n_roi_0} ROI pixels; '
                        f'after {duration:.2f} seconds')

        if seed_output is not None:
            logger.info(f'writing {str(seed_output)}')
            with open(seed_output, 'w') as out_file:
                out_file.write(json.dumps(seed_record, indent=2))

        logger.info(f'writing {str(roi_output)}')
        with open(roi_output, 'w') as out_file:
            out_file.write(json.dumps(list(roi_list), indent=2))

        if plot_output is not None:
            logger.info(f'writing {str(plot_output)}')
            create_roi_plot(plot_output, img_data, roi_list)

        duration = time.time()-t0
        logger.info(f'Completed segmentation in {duration:.2f} seconds')
        return None
