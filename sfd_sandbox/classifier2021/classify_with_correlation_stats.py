import h5py
import json
import numpy as np
import pathlib
import time
import copy

import argparse

import multiprocessing
from ophys_etl.modules.segmentation.utils.multiprocessing_utils import (
    _winnow_process_list)
from ophys_etl.modules.segmentation.utils.stats_utils import (
    estimate_std_from_interquartile_range)


def z_score_of_data(
       roi_data,
       background_data):

    med = np.median(background_data)
    std = estimate_std_from_interquartile_range(background_data)
    z_score = (roi_data-med)/std
    z_score = np.median(z_score)
    return z_score


def corr_from_traces(
    raw_trace0,
    i_trace0,
    raw_trace1,
    i_trace1,
    filter_fraction,
    trace_lookup):

    if i_trace0 not in trace_lookup:
        th = np.quantile(raw_trace0, 1.0-filter_fraction)
        mask0 = np.where(raw_trace0 >= th)[0]
        trace_lookup[i_trace0] = mask0

    if i_trace1 not in trace_lookup:
        th = np.quantile(raw_trace1, 1.0-filter_fraction)
        mask1 = np.where(raw_trace1 >= th)[0]
        trace_lookup[i_trace1] = mask1

    mask0 = trace_lookup[i_trace0]
    mask1 = trace_lookup[i_trace1]

    mask = np.unique(np.concatenate([mask0, mask1]))
    nt = len(mask)

    trace0 = raw_trace0[mask]
    mu0 = np.mean(trace0)
    var0 = np.sum((trace0-mu0)**2)/(nt-1)

    trace1 = raw_trace1[mask]
    mu1 = np.mean(trace1)
    var1 = np.sum((trace1-mu1)**2)/(nt-1)

    num = np.mean((trace0-mu0)*(trace1-mu1))
    denom = np.sqrt(var1*var0)

    return (num/denom, trace_lookup)


def get_roi_v_background_correlation(
        roi_trace_array,
        background_trace_array,
        filter_fraction):

    n_roi = roi_trace_array.shape[0]
    n_bckgd = background_trace_array.shape[0]

    roi_to_roi = np.zeros(n_roi*(n_roi-1)//2, dtype=float)
    roi_to_bckgd = np.zeros(n_roi*n_bckgd, dtype=float)

    trace_lookup = dict()

    ct_roi_to_roi = 0
    ct_roi_to_bckgd = 0
    for ii in range(n_roi):
        raw_roi_0 = roi_trace_array[ii, :]
        for jj in range(ii+1, n_roi, 1):
            raw_roi_1 = roi_trace_array[jj, :]
            (val,
             trace_lookup) = corr_from_traces(
                                 raw_roi_0, ii,
                                 raw_roi_1, jj,
                                 filter_fraction,
                                 trace_lookup)

            roi_to_roi[ct_roi_to_roi] = val
            ct_roi_to_roi += 1

        for jj in range(n_bckgd):
            b_lookup = -1*jj-1
            assert b_lookup < 0
            if ii == 0:
                assert b_lookup not in trace_lookup
            raw_bckgd = background_trace_array[jj, :]
            (val,
             trace_lookup) = corr_from_traces(
                                raw_roi_0, ii,
                                raw_bckgd, -1*jj-1,
                                filter_fraction,
                                trace_lookup)
            roi_to_bckgd[ct_roi_to_bckgd] = val
            ct_roi_to_bckgd += 1

    assert ct_roi_to_roi == n_roi*(n_roi-1)//2
    assert ct_roi_to_bckgd == n_roi*n_bckgd

    rr25, rr75 = np.quantile(roi_to_roi, (0.25, 0.75))
    rb25, rb75 = np.quantile(roi_to_bckgd, (0.25, 0.75))

    z_score = z_score_of_data(
                 roi_data=roi_to_roi,
                 background_data=roi_to_bckgd)

    return {'mean_dcorr_score': np.mean(roi_to_roi)-np.mean(roi_to_bckgd),
            'median_dcorr_score': np.median(roi_to_roi)-np.median(roi_to_bckgd),
            'quantile0.25_dcorr_score': rr25-rb25,
            'quantile0.75_dcorr_score': rr75-rb75,
            'corr_z_score': z_score}


def get_pixel_to_pixel_correlation(
        trace_array,
        filter_fraction):

    trace_lookup = dict()

    n_traces = trace_array.shape[0]
    n_correlations = n_traces*(n_traces-1)//2
    correlations = np.zeros(n_correlations)

    corr_index = 0
    for ii in range(n_traces):
        raw_trace0 = trace_array[ii,:]
        for jj in range(ii+1, n_traces, 1):
            raw_trace1 = trace_array[jj, :]
            (val,
             trace_lookup) = corr_from_traces(
                                 raw_trace0,
                                 ii,
                                 raw_trace1,
                                 jj,
                                 filter_fraction,
                                 trace_lookup)
            correlations[corr_index] = val
            corr_index += 1

    assert corr_index == n_correlations

    c25, c75 = np.quantile(correlations, (0.25, 0.75))

    return {'mean': np.mean(correlations),
            'median': np.median(correlations),
            'quantile0.25': c25,
            'quantile0.75': c75,
            'max': np.max(correlations)}


def get_trace_array_from_roi(video_data, roi):
    if 'mask' in roi:
        mask = roi['mask']
    else:
        mask = roi['mask_matrix']

    mask = np.array(mask)
    n_pixels = np.sum(mask)
    traces = np.zeros((n_pixels, video_data.shape[0]))
    trace_index = 0
    for r in range(roi['height']):
        row = r+roi['y']
        for c in range(roi['width']):
            col = c+roi['x']
            if not mask[r, c]:
                continue
            traces[trace_index, :] = video_data[:, row, col]
            trace_index += 1

    return traces

def get_background_pixels(roi, img_shape):
    if 'mask' in roi:
        mask = roi['mask']
    else:
        mask = roi['mask_matrix']

    mask = np.array(mask)
    n_pixels = np.sum(mask)

    center_row = 0
    center_col = 0
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if not mask[r, c]:
                continue
            center_row += roi['y']+r
            center_col += roi['x']+c

    center_row = np.round(center_row/n_pixels).astype(int)
    center_col = np.round(center_col/n_pixels).astype(int)

    buff = 5
    n_bckgd = 0
    while n_bckgd < n_pixels:
        bckgd_pixels = []
        row0 = max(0,center_row-buff)
        row1 = min(img_shape[0], center_row+buff)
        col0 = max(0,center_col-buff)
        col1 = min(img_shape[1], center_col+buff)
        n_bckgd = 0
        for r in range(row0, row1, 1):
            row = r-roi['y']
            for c in range(col0, col1, 1):
                col = c-roi['x']
                is_bckgd = False
                if row<0 or row>=mask.shape[0]:
                    is_bckgd = True
                elif col<0 or col>=mask.shape[1]:
                    is_bckgd = True
                elif not mask[row, col]:
                    is_bckgd = True

                if is_bckgd:
                    n_bckgd += 1
                    bckgd_pixels.append((r, c))
        buff += 2

    return bckgd_pixels


def get_background_trace_array(video_data, bckgd_pixels):

    n_bckgd = len(bckgd_pixels)

    traces = np.zeros((n_bckgd, video_data.shape[0]))

    for ii in range(n_bckgd):
        pix = bckgd_pixels[ii]
        traces[ii, :] = video_data[:, pix[0], pix[1]]

    return traces


def get_background_fluxes(img_data, background_pixels):
    n_bckgd = len(background_pixels)
    fluxes = np.zeros(n_bckgd, dtype=float)
    for ii in range(n_bckgd):
        pixel = background_pixels[ii]
        fluxes[ii] = img_data[pixel[0], pixel[1]]
    return fluxes

def get_roi_fluxes(img_data, roi):
    if 'mask' in roi:
        mask = roi['mask']
    else:
        mask = roi['mask_matrix']

    mask = np.array(mask)
    n_pixels = np.sum(mask)
    fluxes = np.zeros(n_pixels, dtype=float)
    flux_index = 0
    for r in range(roi['height']):
        row = r+roi['y']
        for c in range(roi['width']):
            col = c+roi['x']
            if not mask[r, c]:
                continue
            fluxes[flux_index] = img_data[row, col]
            flux_index += 1
    return fluxes



def roi_worker(roi_id, trace_array, filter_fraction, output_dict):
    result = get_pixel_to_pixel_correlation(
                    trace_array,
                    filter_fraction)

    output_dict[roi_id] = result


def diff_worker(roi,
                roi_trace_array,
                bckgd_trace_array,
                filter_fraction,
                max_img,
                avg_img,
                background_pixels,
                output_dict):


    roi_id = roi['id']
    assert roi_id not in output_dict

    result = get_roi_v_background_correlation(
                    roi_trace_array,
                    bckgd_trace_array,
                    filter_fraction)


    roi_max = get_roi_fluxes(max_img, roi)
    bckgd_max = get_background_fluxes(max_img, background_pixels)

    max_img_z_score = z_score_of_data(
                            roi_data=roi_max,
                            background_data=bckgd_max)

    roi_avg = get_roi_fluxes(avg_img, roi)
    bckgd_avg = get_background_fluxes(avg_img, background_pixels)

    avg_img_z_score = z_score_of_data(
                            roi_data=roi_avg,
                            background_data=bckgd_avg)


    result['max_img_z_score'] = max_img_z_score
    result['avg_img_z_score'] = avg_img_z_score

    output_dict[roi_id] = result



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--roi_path', type=str, default=None)
    parser.add_argument('--out_path', type=str, default=None)
    parser.add_argument('--filter_fraction', type=float, default=0.05)
    parser.add_argument('--n_processors', type=int, default=8)
    parser.add_argument('--n_roi', type=int, default=-1)
    args = parser.parse_args()

    fontsize=15

    assert args.out_path is not None

    video_path = pathlib.Path(args.video_path)
    roi_path = pathlib.Path(args.roi_path)
    out_path = pathlib.Path(args.out_path)

    assert video_path.is_file()
    assert roi_path.is_file()

    with open(roi_path, 'rb') as in_file:
        roi_list = [roi for roi in json.load(in_file)]

    with h5py.File(video_path, 'r') as in_file:
        video_data = in_file['data'][()]

    max_img = np.max(video_data, axis=0)
    avg_img = np.mean(video_data, axis=0)

    mgr = multiprocessing.Manager()
    roi_corr_dict = mgr.dict()

    if args.n_roi > 0:
        raw_roi_list = copy.deepcopy(roi_list)
        roi_list = [r for r in raw_roi_list[:args.n_roi//2]]
        roi_list += [r for r in raw_roi_list[-1*args.n_roi//2:]]

    n_roi = len(roi_list)
    print(f'n_roi {n_roi}')

    #signals
    t0 = time.time()
    process_list = []
    for i_roi in range(len(roi_list)):
        roi = roi_list[i_roi]
        roi_traces = get_trace_array_from_roi(video_data, roi)

        background_pixels = get_background_pixels(roi, video_data.shape[1:])

        background_traces = get_background_trace_array(
                                    video_data,
                                    background_pixels)

        p = multiprocessing.Process(
                target=diff_worker,
                args=(roi,
                      roi_traces,
                      background_traces,
                      args.filter_fraction,
                      max_img,
                      avg_img,
                      background_pixels,
                      roi_corr_dict))

        p.start()
        process_list.append(p)
        while len(process_list) > 0 and len(process_list) >= args.n_processors:
            process_list = _winnow_process_list(process_list)

        if i_roi > 0 and (i_roi % (2*args.n_processors) == 0) and len(roi_corr_dict) > 0:
            duration = time.time()-t0
            done = len(roi_corr_dict)
            per = duration/done
            pred = per*n_roi
            remaining = pred-duration
            print(f'{done} of {n_roi} signals in {duration:.2e} '
                  f'-- {remaining:.2e} remains '
                  f'of {pred:.2e}')

    print('final batch of signal processes')
    for p in process_list:
        p.join()
    print(f'took {time.time()-t0:.2e} total')
    print(len(roi_corr_dict))
    assert len(roi_corr_dict) == len(roi_list)

    roi_lookup = {roi['id']: roi
                  for roi in roi_list}

    labeled_rois = []
    for roi_id in roi_lookup:
        stats = roi_corr_dict[roi_id]
        roi = roi_lookup[roi_id]
        roi['classifier_scores'] = stats
        labeled_rois.append(roi)

    with open(args.out_path, 'w') as out_file:
        out_file.write(json.dumps(labeled_rois, indent=2))
