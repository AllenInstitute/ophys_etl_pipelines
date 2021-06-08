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
                    img_data: np.ndarray,
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
    fig = figure.Figure(figsize=(60, 20))
    axes = [fig.add_subplot(1, 3, i) for i in [1, 2, 3]]
    axes[0].imshow(img_data)
    axes[1].imshow(img_data)
    axes[2].imshow(img_data)

    for axis, roi_list in zip(axes[1:], (roi_list_0, roi_list_1)):

        bdry_pixels = np.zeros(img_data.shape, dtype=int)
        for roi in roi_list:
            ophys_roi = OphysROI(
                        roi_id=0,
                        x0=roi['x'],
                        y0=roi['y'],
                        width=roi['width'],
                        height=roi['height'],
                        valid_roi=False,
                        mask_matrix=roi['mask'])

            bdry = ophys_roi.boundary_mask
            for ir in range(ophys_roi.height):
                for ic in range(ophys_roi.width):
                    if bdry[ir, ic]:
                        bdry_pixels[ir+ophys_roi.y0,
                                    ic+ophys_roi.x0] = 1

        bdry_pixels = np.ma.masked_where(bdry_pixels == 0,
                                     bdry_pixels)
        axis.imshow(bdry_pixels, cmap='autumn', alpha=1.0)
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
