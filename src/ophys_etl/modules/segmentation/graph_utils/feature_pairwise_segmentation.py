from typing import Tuple, List
import numpy as np
import multiprocessing
import pathlib
import time

from ophys_etl.modules.segmentation.\
graph_utils.feature_vector_segmentation import (
    find_peaks,
    FeatureVectorSegmenter,
    ROISeed)

from ophys_etl.modules.segmentation.graph_utils.feature_vector_rois import (
    PearsonFeatureROI)


def correlate_pixel(pixel_pt: Tuple[int, int],
                    origin: Tuple[int, int],
                    full_img_shape: Tuple[int, int],
                    filter_fraction: float,
                    video_data: np.ndarray,
                    output_dict: multiprocessing.managers.DictProxy):
    """
    pixel_pt is not origin subtracted
    video_data is a subset
    """
    t0 = time.time()
    corrected_pt = (pixel_pt[0]-origin[0],
                    pixel_pt[1]-origin[1])
    sub_img_shape = video_data.shape[1:]
    print(sub_img_shape)
    video_data = video_data.reshape(video_data.shape[0], -1).transpose()
    video_data = video_data.astype(float)
    i_pixel = np.ravel_multi_index(corrected_pt, sub_img_shape)

    n_pixels = video_data.shape[0]

    trace = video_data[i_pixel, :]

    threshold = np.quantile(trace, 1.0-filter_fraction)

    valid_time = (trace>=threshold)
    trace = trace[valid_time]
    video_data = video_data[:, valid_time].astype(float)

    t1 = time.time()

    mu = np.mean(trace)
    video_mu = np.mean(video_data, axis=1)
    assert video_mu.shape == (n_pixels, )

    trace -= mu
    video_data -= mu

    trace_var = np.mean(trace**2)
    video_var = np.mean(video_data**2, axis=1)

    num = np.dot(video_data, trace)
    denom = np.sqrt(trace_var*video_var)
    corr = num/denom
    _d = time.time()-t1
    print(f'corr block took {_d:.2f} seconds; {n_pixels} pixels')
    print(f'{video_data.shape}')

    (img_rows,
     img_cols) = np.meshgrid(np.arange(sub_img_shape[0], dtype=int),
                             np.arange(sub_img_shape[1], dtype=int),
                             indexing='ij')

    img_rows = img_rows.flatten()+origin[0]
    img_cols = img_cols.flatten()+origin[1]

    pixel_indexes = np.ravel_multi_index(np.array([img_rows, img_cols]),
                                         full_img_shape)

    output_dict[pixel_pt] = {'corr':corr,
                             'pixels': pixel_indexes}

    duration = time.time()-t0
    print(f'{pixel_pt} took {duration:.2f} seconds')


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

        self.pre_corr_mgr = multiprocessing.Manager()
        self.pre_corr_lookup = self.pre_corr_mgr.dict()
        self.slop = 20

        super().__init__(graph_input,
                         video_input,
                         attribute=attribute,
                         filter_fraction=filter_fraction,
                         n_processors=n_processors,
                         roi_class=roi_class)


    def _run(self,
             img_data: np.ndarray,
             video_data: np.ndarray) -> List[dict]:


        seed_list = find_peaks(img_data,
                               mask=self.roi_pixels,
                               slop=self.slop)

        img_shape = video_data.shape[1:]

        (img_rows,
         img_cols) = np.meshgrid(np.arange(video_data.shape[1], dtype=int),
                                 np.arange(video_data.shape[2], dtype=int),
                                 indexing='ij')

        img_rows = img_rows.flatten()
        img_cols = img_cols.flatten()

        pt_list = []
        pt_set = set()
        for seed in seed_list:
            center = seed['center']
            rowmin = max(0, center[0]-self.slop)
            rowmax = min(img_shape[0], center[0]+self.slop+1)
            colmin = max(0, center[1]-self.slop)
            colmax = min(img_shape[1], center[1]+self.slop+1)
            for row in range(rowmin, rowmax):
                for col in range(colmin, colmax):
                    pt = (row, col)
                    if pt not in pt_set:
                        pt_set.add(pt)
                        pt_list.append(pt)

        # pre-compute correlations
        t0 = time.time()
        p_list = []
        ct_done = 0
        n_pts = len(pt_list)
        print('need to correlate %d pts' % n_pts)
        print('from %d seeds' % len(seed_list))
        for center in pt_list:
            if center in self.pre_corr_lookup:
                continue

            # factor of 2 is to accommodate case where
            # two pixels are on opposite corners of a tile
            rowmin = max(0, center[0]-2*self.slop)
            rowmax = min(img_shape[0], center[0]+2*self.slop+1)
            colmin = max(0, center[1]-2*self.slop)
            colmax = min(img_shape[1], center[1]+2*self.slop+1)
            origin = (rowmin, colmin)

            sub_video = video_data[:,
                                   rowmin:rowmax,
                                   colmin:colmax]


            args = (center,
                    origin,
                    img_shape,
                    self._filter_fraction,
                    sub_video,
                    self.pre_corr_lookup)


            correlate_pixel(*args)
            exit()

            p = multiprocessing.Process(target=correlate_pixel,
                                        args=args)
            p.start()
            p_list.append(p)
            # make sure that all processors are working at all times,
            # if possible
            while len(p_list) > 0 and len(p_list) >= self.n_processors-1:
                to_pop = []
                for ii in range(len(p_list)-1, -1, -1):
                    if p_list[ii].exitcode is not None:
                        to_pop.append(ii)
                for ii in to_pop:
                    p_list.pop(ii)
                    ct_done +=1
                    if ct_done % 20 == 0:
                        duration = time.time()-t0
                        print(f'{ct_done} out of {n_pts} in {duration:.2f} sec')

        for p in p_list:
            p.join()

        duration = time.time()-t0
        print('pre correlation took %e seconds' % duration)
        time.sleep(30)
