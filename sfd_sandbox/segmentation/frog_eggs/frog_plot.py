import matplotlib.figure as figure
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


def create_roi_plot(plot_path: pathlib.Path,
                    raw_img_data: np.ndarray,
                    roi_list_0: List[dict],
                    roi_list_1: List[dict]) -> None:
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

    fig = figure.Figure(figsize=(60, 20))
    axes = [fig.add_subplot(1, 3, i) for i in [1, 2, 3]]
    img = axes[0].imshow(rgb_img_data)
    #axes[1].imshow(rgb_img_data)
    #axes[2].imshow(rgb_img_data)

    alpha = 0.75
    color_list = ((255, 0, 0),
                  (0, 255, 0),
                  (0, 0, 255),
                  (125, 0, 125),
                  (0, 100, 50))

    for axis, roi_list in zip(axes[1:], (roi_list_0, roi_list_1)):

        img_data = np.copy(rgb_img_data)
        color_index = 0

        for roi in roi_list:
            color = color_list[color_index]
            color_index += 1
            if color_index >= len(color_list):
                color_index = 0
            ophys_roi = OphysROI(
                        roi_id=0,
                        x0=roi['x'],
                        y0=roi['y'],
                        width=roi['width'],
                        height=roi['height'],
                        valid_roi=False,
                        mask_matrix=roi['mask'])

            bdry = ophys_roi.mask_matrix
            for ir in range(ophys_roi.height):
                for ic in range(ophys_roi.width):
                    if bdry[ir, ic]:
                        row = ir+ophys_roi.y0
                        col = ic+ophys_roi.x0
                        old = img_data[row, col, :]
                        for i_color in range(3):
                            new = old[i_color]*(1.0-alpha)+color[i_color]*alpha
                            new = np.uint8(min(new,255))
                            img_data[row, col, i_color] = new

        axis.imshow(img_data)
    fig.tight_layout()
    fig.savefig(plot_path)
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', type=str, default=None)
    parser.add_argument('--roi0', type=str, default=None)
    parser.add_argument('--roi1', type=str, default=None)
    parser.add_argument('--plot', type=str, default=None)
    args = parser.parse_args()

    graph = nx.read_gpickle(args.graph)
    with open(args.roi0,'rb') as in_file:
        roi0 = json.load(in_file)
    with open(args.roi1,'rb') as in_file:
        roi1 = json.load(in_file)
    create_roi_plot(pathlib.Path(args.plot),
                                 graph_to_img(graph),roi0,roi1)
