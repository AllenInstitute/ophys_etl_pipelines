import matplotlib.figure as figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import argparse
from typing import List

from ophys_etl.modules.segmentation.graph_utils.plotting import (
    graph_to_img,
    create_roi_plot)

import networkx as nx
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
import json
import pathlib
import ophys_etl.modules.segmentation.postprocess_utils.roi_merging as merging


def to_rgb(raw_img_data):
    rgb_img_data = np.zeros((raw_img_data.shape[0],
                             raw_img_data.shape[1],
                             3),dtype=np.uint8)

    mx = raw_img_data.max()
    img_data = np.round(255*raw_img_data.astype(float)/mx).astype(np.uint8)
    for ic in range(3):
        rgb_img_data[:,:,ic] = img_data
    return rgb_img_data


def add_bdry_to_rgb(img_data, ophys_roi,
                    color=(255, 128, 0), alpha=0.75):
    bdry = ophys_roi.boundary_mask
    for ir in range(ophys_roi.height):
        for ic in range(ophys_roi.width):
            if bdry[ir, ic]:
                row = ir+ophys_roi.y0
                col = ic+ophys_roi.x0
                old = img_data[row, col, :]
                for i_color in range(3):
                    new = alpha*color[i_color]+(1.0-alpha)*old[i_color]
                    new = np.uint8(new)
                    img_data[row, col, i_color] = new

    return img_data


def get_inactive_mask(img_data_shape, roi_list):
    full_mask = np.zeros(img_data_shape, dtype=bool)
    for roi in roi_list:
        mask = roi.mask_matrix
        region = full_mask[roi.y0:roi.y0+roi.height,
                           roi.x0:roi.x0+roi.width]
        full_mask[roi.y0:roi.y0+roi.height,
                  roi.x0:roi.x0+roi.width] = np.logical_or(mask, region)
    print('full mask ',full_mask.min(),full_mask.max())
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


def get_nsigma_img(img_data, roi_list, dx=10):
    final_img = np.zeros(img_data.shape, dtype=float)
    inactive_mask = get_inactive_mask(img_data.shape, roi_list)
    for roi in roi_list:
        mu, std = get_inactive_dist(img_data, roi, inactive_mask, dx=dx)
        mask = roi.mask_matrix
        xmin = roi.x0
        xmax = xmin+roi.width
        ymin = roi.y0
        ymax = ymin+roi.height
        roi_pixels = img_data[ymin:ymax, xmin:xmax][mask].flatten()
        n_sigma = np.mean((roi_pixels-mu)/std)
        final_img[ymin:ymax, xmin:xmax][mask] = n_sigma
    return final_img


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default=None)
    parser.add_argument('--roi', type=str, default=None)
    parser.add_argument('--plot', type=str, default=None)
    args = parser.parse_args()

    fig = figure.Figure(figsize=(20, 30))
    axes = [fig.add_subplot(3,2,i) for i in (1,2,3,4,5,6)]

    for ii in (5,):
        axes[ii].tick_params(left=0, bottom=0,
                             labelleft=0, labelbottom=0)
        for s in ('top', 'bottom', 'left', 'right'):
            axes[ii].spines[s].set_visible(False)

    graph = nx.read_gpickle(args.graph)
    img = graph_to_img(graph)
    rgb_img = to_rgb(img)
    axes[0].imshow(rgb_img)

    with open(args.roi, 'rb') as in_file:
        raw_roi = json.load(in_file)
    roi_list = []
    for roi in raw_roi:
        new_roi = merging.extract_roi_to_ophys_roi(roi)
        roi_list.append(new_roi)
    bdry_img = np.copy(rgb_img)
    for roi in roi_list:
        bdry_img = add_bdry_to_rgb(rgb_img, roi)
    axes[1].imshow(bdry_img)

    inactive_mask = get_inactive_mask(img.shape, roi_list)
    shown_img = axes[2].imshow(inactive_mask)
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(shown_img, ax=axes[2], cax=cax)

    nsigma_img = get_nsigma_img(img, roi_list)
    nsigma_img_cut = np.where(nsigma_img<10,nsigma_img,10)
    shown_img = axes[3].imshow(nsigma_img_cut)
    divider = make_axes_locatable(axes[3])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(shown_img, ax=axes[3], cax=cax)

    mask = (nsigma_img>=10)
    axes[4].imshow(mask)

    fig.tight_layout()
    fig.savefig(args.plot)

    new = np.where(nsigma_img < 100, nsigma_img, 100)
    alt_fig = figure.Figure(figsize=(50,50))
    ax = alt_fig.add_subplot(1,1,1)
    shown_img = ax.imshow(new)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    alt_fig.colorbar(shown_img, ax=ax, cax=cax)
    cax.tick_params(labelsize=45)
    alt_fig.savefig('alt.png')
