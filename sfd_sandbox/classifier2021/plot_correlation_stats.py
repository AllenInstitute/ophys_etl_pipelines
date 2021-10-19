import matplotlib.figure as mplt_fig
import matplotlib.colors as mplt_colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

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


def get_background_trace_array_from_roi(video_data, roi):
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
        row1 = min(video_data.shape[1], center_row+buff)
        col0 = max(0,center_col-buff)
        col1 = min(video_data.shape[2], center_col+buff)
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

    traces = np.zeros((n_bckgd, video_data.shape[0]))
    for ii in range(n_bckgd):
        pix = bckgd_pixels[ii]
        traces[ii, :] = video_data[:, pix[0], pix[1]]

    return traces


def roi_worker(roi_id, trace_array, filter_fraction, output_dict):
    result = get_pixel_to_pixel_correlation(
                    trace_array,
                    filter_fraction)

    output_dict[roi_id] = result


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default=None)
    parser.add_argument('--roi_path', type=str, default=None)
    parser.add_argument('--min_score', type=float, default=0.3)
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
        roi_list = [roi for roi in json.load(in_file)
                    if args.min_score < 0.0
                    or roi['classifier_score'] >= args.min_score]

    with h5py.File(video_path, 'r') as in_file:
        video_data = in_file['data'][()]

    mgr = multiprocessing.Manager()
    roi_corr_dict = mgr.dict()
    roi_bckgd_dict = mgr.dict()

    if args.n_roi > 0:
        raw_roi_list = copy.deepcopy(roi_list)
        roi_list = [r for r in raw_roi_list[:args.n_roi//2]]
        roi_list += [r for r in raw_roi_list[-1*args.n_roi//2:]]

    n_roi = len(roi_list)
    print(f'min score {args.min_score}')
    print(f'n_roi {n_roi}')

    # backgrounds
    t0 = time.time()
    process_list = []
    for i_roi in range(len(roi_list)):
        roi = roi_list[i_roi]
        traces = get_background_trace_array_from_roi(video_data, roi)
        p = multiprocessing.Process(
                target=roi_worker,
                args=(roi['id'],
                      traces,
                      args.filter_fraction,
                      roi_bckgd_dict))

        p.start()
        process_list.append(p)
        while len(process_list) > 0 and len(process_list) >= args.n_processors:
            process_list = _winnow_process_list(process_list)

        if i_roi > 0 and i_roi %5 == 0 and len(roi_bckgd_dict) > 0:
            duration = time.time()-t0
            done = len(roi_bckgd_dict)
            per = duration/done
            pred = per*n_roi
            remaining = pred-duration
            print(f'{done} of {n_roi} bckgds in {duration:.2e} '
                  f'-- {remaining:.2e} remains '
                  f'of {pred:.2e}')

    print('final batch of bckgd processes')
    for p in process_list:
        p.join()
    print(f'took {time.time()-t0:.2e} total')
    print(len(roi_bckgd_dict))
    assert len(roi_bckgd_dict) == len(roi_list)

    #signals
    t0 = time.time()
    process_list = []
    for i_roi in range(len(roi_list)):
        roi = roi_list[i_roi]
        traces = get_trace_array_from_roi(video_data, roi)
        p = multiprocessing.Process(
                target=roi_worker,
                args=(roi['id'],
                      traces,
                      args.filter_fraction,
                      roi_corr_dict))

        p.start()
        process_list.append(p)
        while len(process_list) > 0 and len(process_list) >= args.n_processors:
            process_list = _winnow_process_list(process_list)

        if i_roi > 0 and i_roi %5 == 0 and len(roi_corr_dict) > 0:
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

    blank = -999.0*np.ones((512, 512), dtype=float)

    fig = mplt_fig.Figure(figsize=(10, 10))
    axes = [fig.add_subplot(2,2,ii) for ii in range(1,5)]
    for axis, stat in zip(axes, ('mean', 'median',
                                 'quantile0.25', 'quantile0.75')):
        img = np.copy(blank)
        for roi in roi_list:
            val = roi_corr_dict[roi['id']][stat]-roi_bckgd_dict[roi['id']][stat]
            if 'mask' in roi:
                mask = roi['mask']
            else:
                mask = roi['mask_matrix']
            for r in range(len(mask)):
                row = r+roi['y']
                for c in range(len(mask[0])):
                    col = c+roi['x']
                    if not mask[r][c]:
                        continue
                    img[row, col] = val

        img = np.ma.masked_where(img<-900.0, img)

        plot_img = axis.imshow(
                      img,
                      cmap='jet_r',
                      vmin=0.0,
                      vmax=1.0)
                      #norm=mplt_colors.LogNorm(vmin=args.vmin, vmax=1.0),

        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(plot_img, cax=cax)
        axis.set_title(stat, fontsize=fontsize)

    for ax in axes:
        for ii in range(128, 512, 128):
            ax.axhline(ii, color='k', alpha=0.5)
            ax.axvline(ii, color='k', alpha=0.5)

    fig.tight_layout()
    fig.savefig(args.out_path)
