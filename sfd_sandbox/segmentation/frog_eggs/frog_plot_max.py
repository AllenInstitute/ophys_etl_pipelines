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

def select_color(this_roi, raw_plotted_rois, color_list, roi_to_color):
    neighbor_colors = []
    neighbor_color_sizes = []

    centroid_x = np.array([r.centroid_x for r in raw_plotted_rois])
    centroid_y = np.array([r.centroid_y for r in raw_plotted_rois])
    dist = np.sqrt((this_roi.centroid_x-centroid_x)**2 +
                   (this_roi.centroid_y-centroid_y)**2)

    valid = dist<10
    plotted_rois = np.array(raw_plotted_rois)[valid]
    for roi in plotted_rois:
        if merging.do_rois_abut(this_roi, roi, dpix=3):
            neighbor_colors.append(roi_to_color[roi.roi_id])
            neighbor_color_sizes.append(roi.mask_matrix.sum())
    if len(neighbor_colors) == 0:
        return color_list[0]

    if len(set(neighbor_colors)) < len(color_list):
        neighbor_colors = set(neighbor_colors)
        for color in color_list:
            if color not in neighbor_colors:
                return color

    neighbor_color_sizes = np.array(neighbor_color_sizes)
    sorted_dex = np.argsort(-1*neighbor_color_sizes)
    unq_color_set = set()
    for ii in sorted_dex:
        if len(unq_color_set) == len(color_list)-1:
            return neighbor_colors[ii]
        unq_color_set.add(neighbor_colors[ii])
        print(neighbor_colors[ii],unq_color_set)

    print(len(neighbor_colors))
    print(len(color_list))
    print(len(unq_color_set))
    raise RuntimeError("should not be here")

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

        plotted_rois = []
        roi_to_color = {}
        ophys_roi_list = []
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

        print(f'{len(ophys_roi_list)} rois')
        ct = 0
        for ophys_roi in ophys_roi_list:
            ct += 1
            if ct%100 ==0:
                print(ct)
            color = select_color(ophys_roi,
                                 plotted_rois,
                                 color_list,
                                 roi_to_color)

            plotted_rois.append(ophys_roi)
            roi_to_color[ophys_roi.roi_id] = color

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

    for exp_id in experiment_ids[:1]:
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
