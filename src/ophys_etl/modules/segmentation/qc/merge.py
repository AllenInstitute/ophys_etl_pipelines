import copy
import numpy as np
import networkx as nx
from matplotlib import cm
from matplotlib.figure import Figure
from typing import List

from ophys_etl.modules.segmentation.qc_utils import roi_utils
from ophys_etl.modules.segmentation.utils.roi_utils import convert_roi_keys
from ophys_etl.types import ExtractROI


def roi_ancestor_gallery(figure: Figure,
                         metric_image: np.ndarray,
                         attribute: str,
                         original_roi_list: List[ExtractROI],
                         merged_roi_list: List[ExtractROI],
                         merger_ids: np.ndarray,
                         merged_id: int) -> None:
    """creates a figure that shows an ROI and all its pre-merge ancestors

    Parameters
    ----------
    figure: matplotlib.figure.Figure
        the figure to write into
    metric_image: np.ndarray
        the metric to plot and use for calculating average metric
    attribute: str
        the name of the metric, displayed in a plot title
    original_roi_list: List[ExtractROI]
        the unmerged ROI list
    merged_roi_list: List[ExtractROI]
        the merged ROI list
    merger_ids: np.ndarray
        n_merge x 2 array, each row listing the IDs of a single merge.
    merged_id: int
        the ID in the merged list for which to show ancestors

    """

    # create a graph to separate ancestors
    graph = nx.from_edgelist(merger_ids)
    subgraphs = list(nx.connected_components(graph))
    subgraph = None

    for s in subgraphs:
        if merged_id in s:
            subgraph = s

    # if uninvolved in mergers, ancestor is itself
    if subgraph is None:
        subgraph = {merged_id}

    # approximate grid for number of plots
    n_plots = len(subgraph) + 1
    n_row = int(np.ceil(np.sqrt(n_plots)))
    n_col = n_row
    while (n_row * n_col - n_plots) > n_col:
        n_row -= 1
    plot_index = 1

    # merged ROI
    a0 = figure.add_subplot(n_row, n_col, plot_index)
    a0.imshow(metric_image)
    merged = [i for i in merged_roi_list if i['id'] == merged_id]
    roi_utils.add_rois_to_axes(a0,
                               merged,
                               metric_image.shape)
    a0.set_title(f"metric: {attribute}\nmerged ROI {merged_id}")

    axes = []
    # ancestor ROIs
    for ancestor in subgraph:
        plot_index += 1
        roi = [i for i in original_roi_list if i['id'] == ancestor]
        axes.append(
                figure.add_subplot(n_row, n_col, plot_index,
                                   sharex=a0, sharey=a0))
        axes[-1].imshow(metric_image)
        roi_utils.add_rois_to_axes(axes[-1],
                                   roi,
                                   metric_image.shape)
        axes[-1].set_title(f"ancestor ROI {ancestor}")

    figure.tight_layout()

    
def roi_merge_plot(figure: Figure,
                   metric_image: np.ndarray,
                   attribute: str,
                   original_roi_list: List[ExtractROI],
                   merged_roi_list: List[ExtractROI],
                   merger_ids: np.ndarray,
                   seed: int = 32) -> None:
    """creates a QC plot showing which ROIs were merged

    Parameters
    ----------
    figure: matplotlib.figure.Figure
        the figure to write into
    metric_image: np.ndarray
        the metric to plot and use for calculating average metric
    attribute: str
        the name of the metric, displayed in a plot title
    original_roi_list: List[ExtractROI]
        the unmerged ROI list
    merged_roi_list: List[ExtractROI]
        the merged ROI list
    merger_ids: np.ndarray
        n_merge x 2 array, each row listing the IDs of a single merge.
    seed: int
        seed for random generation of colors for mergers

    """

    # separate out mergers with different resulting ROIs
    graph = nx.from_edgelist(merger_ids)
    subgraphs = list(nx.connected_components(graph))

    # define colors for the different mergers
    cmap = cm.plasma
    color_indices = np.linspace(start=0, stop=cmap.N, num=len(subgraphs), dtype=int)
    rng = np.random.default_rng(seed)
    rng.shuffle(color_indices)

    ax00 = figure.add_subplot(1, 2, 1)
    ax01 = figure.add_subplot(1, 2, 2)
    ax00.imshow(metric_image, cmap="gray")
    ax01.imshow(metric_image, cmap="gray")

    for subgraph, color_index in zip(subgraphs, color_indices):
        originals = [i for i in original_roi_list
                     if i["id"] in subgraph]
        merged = [i for i in merged_roi_list
                  if i["id"] in subgraph]
        if len(merged) != 1:
            raise ValueError("Expected a single ROI in merging graph "
                             f"{subgraph} to be in the merged ROI list "
                             f"but got {len(merged)} ROIs: {merged}")
        color = cmap(color_index)
        roi_utils.add_rois_to_axes(ax00,
                                   originals,
                                   metric_image.shape,
                                   rgba=color)
        roi_utils.add_rois_to_axes(ax01,
                                   merged,
                                   metric_image.shape,
                                   rgba=color)

    # plot in gray ROIs that were not part of merging
    untouched = [i for i in original_roi_list if i["id"] not in graph]
    for ax in [ax00, ax01]:
        roi_utils.add_rois_to_axes(ax,
                                   untouched,
                                   metric_image.shape,
                                   rgba=(0.5, 0.5, 0.5, 1.0))
    ax00.set_title(f"metric: {attribute}\nunmerged")
    ax01.set_title(f"metric: {attribute}\nmerged")
