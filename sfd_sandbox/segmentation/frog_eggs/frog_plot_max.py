import matplotlib.figure as figure
import numpy as np
import argparse
from typing import List
import PIL.Image

from ophys_etl.modules.segmentation.graph_utils.plotting import (
    graph_to_img,
    create_roi_plot)

import networkx as nx
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
import json
import pathlib
import ophys_etl.modules.segmentation.postprocess_utils.roi_merging as merging


def create_roi_plot(plot_path: pathlib.Path,
                    raw_img_data: np.ndarray,
                    max_img_data: np.ndarray,
                    roi_list_0: List[dict],
                    roi_list_1: List[dict],
                    seeds=None) -> None:
    """
    Generate a side-by-side plot comparing the image data
    used to seed ROI generation with the borders of the
    discovered ROIs

    Parameters
    ----------
    plot_path: pathlib.Path
        Path to file where plot will be saved

    img_data: np.ndarray
        The baseline image over which to plot the ROIs

    roi_list: List[ExtractROI]

    Returns
    -------
    None
    """

    rgb_img_data = np.zeros((raw_img_data.shape[0],
                             raw_img_data.shape[1],
                             3),dtype=np.uint8)

    mx = raw_img_data.max()
    img_data = np.round(255*raw_img_data.astype(float)/mx).astype(np.uint8)
    for ic in range(3):
        rgb_img_data[:,:,ic] = img_data

    rgb_max_data = np.zeros((raw_img_data.shape[0],
                             raw_img_data.shape[1],
                             3),dtype=np.uint8)

    mx = max_img_data.max()
    max_img_data = np.round(255*max_img_data.astype(float)/mx).astype(np.uint8)
    for ic in range(3):
        rgb_max_data[:,:,ic] = max_img_data

    fig = figure.Figure(figsize=(30, 30))
    axes = [fig.add_subplot(2, 2, i) for i in [1, 2, 3, 4]]
    img = axes[0].imshow(rgb_max_data)
    img = axes[1].imshow(rgb_img_data)
    #axes[1].imshow(rgb_img_data)
    #axes[2].imshow(rgb_img_data)

    alpha = 0.5

    color_list = ((255, 0, 0),
                  (0, 255, 0),
                  (0, 0, 255))

    bdry_color = (255, 128, 0)

    for axis, raw_roi_list, seed_set in zip(axes[2:],
                                            (roi_list_0, roi_list_1),
                                            (None, seeds)):

        img_data = np.copy(rgb_img_data)
        color_index = 0

        ophys_roi_list = []
        centroids = np.zeros((len(raw_roi_list), 2), dtype=float)
        i_roi = 0
        for roi in raw_roi_list:
            ophys_roi = OphysROI(
                        roi_id=0,
                        x0=roi['x'],
                        y0=roi['y'],
                        width=roi['width'],
                        height=roi['height'],
                        valid_roi=False,
                        mask_matrix=roi['mask'])
            ophys_roi_list.append(ophys_roi)
            centroids[i_roi,0] = ophys_roi.centroid_y
            centroids[i_roi,1] = ophys_roi.centroid_x
            i_roi += 1

        print(f'{len(ophys_roi_list)} rois')
        last_row = 0
        last_col = 0
        plotted = 0
        while plotted < len(ophys_roi_list):
            dist = (centroids[:,0]-last_row)**2+(centroids[:,1]-last_col)**2
            chosen = np.argmin(dist)
            ophys_roi = ophys_roi_list[chosen]
            last_row = centroids[chosen, 0]
            last_col = centroids[chosen, 1]
            centroids[chosen, 0] = -9999
            centroids[chosen, 1] = -9999
            plotted += 1

            color = color_list[color_index]
            color_index += 1
            if color_index >= len(color_list):
                color_index = 0

            msk = ophys_roi.mask_matrix
            for ir in range(ophys_roi.height):
                for ic in range(ophys_roi.width):
                    if msk[ir, ic]:
                        row = ir+ophys_roi.y0
                        col = ic+ophys_roi.x0
                        old = img_data[row, col, :]
                        for i_color in range(3):
                            new = alpha*color[i_color]+(1.0-alpha)*img_data[row, col, i_color]
                            new = np.round(new).astype(np.uint8)
                            img_data[row, col, i_color] = new

        for ophys_roi in ophys_roi_list:
            bdry = ophys_roi.boundary_mask
            for ir in range(ophys_roi.height):
                for ic in range(ophys_roi.width):
                    if bdry[ir, ic]:
                        row = ir+ophys_roi.y0
                        col = ic+ophys_roi.x0
                        old = img_data[row, col, :]
                        for i_color in range(3):
                            new = bdry_color[i_color]
                            img_data[row, col, i_color] = new


        axis.imshow(img_data)
        if seed_set is not None:
            roi_list = [merging.extract_roi_to_ophys_roi(r) for r in seed_set]
            xx = [r.centroid_x for r in roi_list]
            yy = [r.centroid_y for r in roi_list]
            axis.scatter(xx, yy, marker='+', color='w', s=200)

    for ax in axes:
        for ix in range(0,512,128):
            ax.axhline(ix,color='w',alpha=0.25)
            ax.axvline(ix,color='w',alpha=0.25)

    fig.tight_layout()
    fig.savefig(plot_path)
    return None


def do_plot(max_img_fname, graph_fname, roi0_fname, roi1_fname, plot_fname):

    graph = nx.read_gpickle(graph_fname)
    with open(roi0_fname,'rb') as in_file:
        roi0 = json.load(in_file)
    with open(roi1_fname,'rb') as in_file:
        roi1 = json.load(in_file)
    seeds = None

    max_img = PIL.Image.open(max_img_fname, 'r')
    max_img = np.array(max_img)

    create_roi_plot(plot_fname,
                    graph_to_img(graph),
                    max_img,
                    roi0,roi1,seeds=seeds)

if __name__ == "__main__":
    experiment_ids=(785569470,
                1048483611,
                1048483613,
                1048483616,
                785569447,
                788422859,
                795901850,
                788422825,
                795897800,
                850517348,
                951980473,
                951980484,
                795901895,
                806862946,
                803965468,
                806928824)

    dk = pathlib.Path('/allen/programs/braintv/workgroups/nc-ophys/danielk')
    assert dk.is_dir()
    expdir = dk /'deepinterpolation/experiments'
    assert expdir.is_dir()
    sfd = pathlib.Path('/allen/aibs/informatics/danielsf/frog_eggs/topography')

    for exp_id in experiment_ids:
        this_dir = expdir/f'ophys_experiment_{exp_id}'
        assert this_dir.is_dir()
        background = this_dir/'backgrounds'
        print(background)
        assert background.is_dir()
        rois = this_dir/'rois'
        assert rois.is_dir()

        max_fname = background/'denoised_maxp.png'
        assert max_fname.is_file()
        roi_fname = rois/'deep_denoised_filtered_hnc_Gaussian_feature_vector_rois.json'
        assert roi_fname.is_file()
        merged_fname = sfd/f'{exp_id}_rois.json'
        assert merged_fname.is_file()

        graph_fname = background/'deep_denoised_filtered_hnc_Gaussian_graph.pkl'
        assert graph_fname.is_file()

        plot_fname = sfd/f'{exp_id}_merge_output.png'
        do_plot(max_fname, graph_fname, roi_fname, merged_fname, plot_fname)
