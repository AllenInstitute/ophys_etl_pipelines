from typing import Tuple, List
import numpy as np
import multiprocessing
import pathlib
import time
import h5py

from ophys_etl.modules.segmentation.\
graph_utils.feature_vector_segmentation import (
    find_peaks,
    FeatureVectorSegmenter,
    ROISeed)

from ophys_etl.modules.segmentation.graph_utils.feature_vector_rois import (
    PearsonFeatureROI)


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
    t0 = time.time()

    n_pixels = video_data.shape[0]
    assert n_pixels == len(pixel_indexes)

    i_pixel = np.where(pixel_indexes==i_pixel_global)[0][0]

    trace = video_data[i_pixel, :]
    threshold = np.quantile(trace, 1.0-filter_fraction)

    valid_time = (trace>=threshold)
    video_data = video_data[:, valid_time].astype(float)
    trace = video_data[i_pixel, :]

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
    #print(f'corr block took {_d:.2f} seconds; {n_pixels} pixels')
    #print(f'{video_data.shape}')
    output_dict[pixel_pt] = {'corr':corr,
                             'pixels': pixel_indexes}

    duration = time.time()-t0
    #print(f'{pixel_pt} took {duration:.2f} seconds')


def correlate_tile(video_path: pathlib.Path,
                   rowbounds: Tuple[int, int],
                   colbounds: Tuple[int, int],
                   img_shape: Tuple[int, int],
                   slop: int,
                   filter_fraction: float,
                   output_dict: multiprocessing.managers.DictProxy):

    local_output_dict = {}

    slop = slop*2

    t0 = time.time()
    n_tot = (rowbounds[1]-rowbounds[0])*(colbounds[1]-colbounds[0])
    rowmin = max(0, rowbounds[0]-slop)
    rowmax = min(img_shape[0], rowbounds[1]+slop+1)
    colmin = max(0, colbounds[0]-slop)
    colmax = min(img_shape[1], colbounds[1]+slop+1)
    with h5py.File(video_path, 'r') as in_file:
        video_data = in_file['data'][:,rowmin:rowmax, colmin:colmax]

    print('\nactually read in video ',video_data.shape)
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

    ct = 0
    for row in range(rowbounds[0], rowbounds[1], 1):
        for col in range(colbounds[0], colbounds[1], 1):
            pixel = (row, col)
            i_pixel = np.ravel_multi_index(pixel, img_shape)

            r0 = max(0, row-slop)
            r1 = min(img_shape[0], row+slop+1)
            c0 = max(0, col-slop)
            c1 = min(img_shape[1], col+slop+1)

            (img_rows,
             img_cols) = np.meshgrid(np.arange(r0-rowmin, r1-rowmin, 1, dtype=int),
                                     np.arange(c0-colmin, c1-colmin, 1, dtype=int),
                                     indexing='ij')

            local_indexes = np.ravel_multi_index([img_rows.flatten(),
                                                  img_cols.flatten()],
                                                 (rowmax-rowmin, colmax-colmin))

            chosen_pixels = pixel_indexes[local_indexes]
            sub_video = video_data[local_indexes, :]
            #print('local_indexes ',len(local_indexes))
            #print(r0,r1,c0,c1)
            #print(r0-rowmin,r1-rowmin,c0-colmin,c1-colmin)

            args = (pixel,
                    i_pixel,
                    filter_fraction,
                    sub_video,
                    chosen_pixels,
                    local_output_dict)

            correlate_pixel(*args)
            ct +=1
            if ct % 100 == 0:
                dur = (time.time()-t0)/3600.0
                per = dur/ct
                pred = per*n_tot
                print(f'{ct} pixels of {n_tot} in {dur:.2f}; {per:.2f}; {pred:.2f} (hrs)')

    # copy results over to output_dict
    key_list = list(local_output_dict.keys())
    for key in key_list:
        obj = local_output_dict.pop(key)
        output_dict[key] = obj


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

    def _load_video(self, video_path:pathlib.Path):
        return np.zeros(3, dtype=float)
        with h5py.File(video_path, 'r') as in_file:
            video_data = in_file['data'][()]
        t0 = time.time()
        video_data = video_data.reshape(video_data.shape[0], -1)
        print(f'reshape {time.time()-t0:.2f}')
        video_data = video_data.transpose()
        print(f'transpose {time.time()-t0:.2f}')
        return video_data

    def _run(self,
             img_data: np.ndarray,
             video_data: np.ndarray) -> List[dict]:
        """
        video data must already be reshaped etc.
        """
        # pre-compute correlations
        img_shape = img_data.shape
        sqrt_proc = 2*np.ceil(np.sqrt(self.n_processors)).astype(int)
        drow = max(1, img_shape[0]//sqrt_proc)
        dcol = max(1, img_shape[1]//sqrt_proc)

        t0 = time.time()
        p_list = []
        ct_done = 0
        for row0 in range(0, img_shape[0], drow):
            row1 = min(img_shape[0], row0+drow)
            for col0 in range(0, img_shape[1], dcol):
                col1 = min(img_shape[1], col0+dcol)

                args = (self._video_input,
                        (row0, row1),
                        (col0, col1),
                        img_shape,
                        self.slop,
                        self._filter_fraction,
                        self.pre_corr_lookup)

                #correlate_tile(*args)


                #exit()

                p = multiprocessing.Process(target=correlate_tile,
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
