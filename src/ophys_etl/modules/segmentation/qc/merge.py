import numpy as np
import networkx as nx
from matplotlib.figure import Figure
from typing import List

from ophys_etl.modules.segmentation.qc_utils import roi_utils

from ophys_etl.modules.segmentation.qc_utils.roi_comparison_utils import (
    get_roi_color_map)

from ophys_etl.types import ExtractROI
from ophys_etl.modules.segmentation.utils.roi_utils import (
    extract_roi_to_ophys_roi)


def roi_ancestor_gallery(figure: Figure,
                         metric_image: np.ndarray,
                         attribute: str,
                         original_roi_list: List[ExtractROI],
                         merged_roi_list: List[ExtractROI],
                         merger_ids: np.ndarray,
                         merged_id: int,
                         full_fov: bool = False,
                         plot_buffer: int = 5) -> None:
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
    full_fov: bool
        whether to display the ROIs and ancestors in the full FOV (true)
        or to zoom to a bounding box around the merged ROI
    plot_buffer: int
        if full_fov is False, adds a buffer to the zoomed plot around
        the merged ROI bounding box

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

    if not full_fov:
        xlim = (max(0, merged[0]["x"] - plot_buffer),
                min(merged[0]["x"] + merged[0]["width"] + plot_buffer,
                    metric_image.shape[1]))
        ylim = (max(0, merged[0]["y"] - plot_buffer),
                min(merged[0]["y"] + merged[0]["height"] + plot_buffer,
                    metric_image.shape[0]))
        a0.set_xlim(xlim)
        a0.set_ylim(ylim[::-1])

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

    dst_id_set = set()
    for pair in merger_ids:
        dst_id_set.add(pair[0])

    rois_for_color_map = []
    for roi in merged_roi_list:
        if roi['id'] in dst_id_set:
            rois_for_color_map.append(extract_roi_to_ophys_roi(roi))

    roi_color_map = get_roi_color_map(rois_for_color_map)

    # separate out mergers with different resulting ROIs
    graph = nx.from_edgelist(merger_ids)
    subgraphs = list(nx.connected_components(graph))

    ax00 = figure.add_subplot(1, 2, 1)
    ax01 = figure.add_subplot(1, 2, 2)
    ax00.imshow(metric_image, cmap="gray")
    ax01.imshow(metric_image, cmap="gray")

    for subgraph in subgraphs:
        originals = [i for i in original_roi_list
                     if i["id"] in subgraph]
        merged = [i for i in merged_roi_list
                  if i["id"] in subgraph]
        if len(merged) != 1:
            raise ValueError("Expected a single ROI in merging graph "
                             f"{subgraph} to be in the merged ROI list "
                             f"but got {len(merged)} ROIs: {merged}")
        rgb_color = roi_color_map[merged[0]['id']]
        rgba_color = (rgb_color[0]/255.0,
                      rgb_color[1]/255.0,
                      rgb_color[2]/255.0,
                      1.0)

        roi_utils.add_rois_to_axes(ax00,
                                   originals,
                                   metric_image.shape,
                                   rgba=rgba_color)
        roi_utils.add_rois_to_axes(ax01,
                                   merged,
                                   metric_image.shape,
                                   rgba=rgba_color)

    # plot in gray ROIs that were not part of merging
    untouched = [i for i in original_roi_list if i["id"] not in graph]
    for ax in [ax00, ax01]:
        roi_utils.add_rois_to_axes(ax,
                                   untouched,
                                   metric_image.shape,
                                   rgba=(0.5, 0.5, 0.5, 1.0))
    ax00.set_title(f"metric: {attribute}\nunmerged")
    ax01.set_title(f"metric: {attribute}\nmerged")
