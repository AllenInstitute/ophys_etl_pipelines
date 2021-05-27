import networkx as nx
import numpy as np
from typing import List
import pathlib
from matplotlib import figure, axes
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI


def draw_graph_edges(figure: figure.Figure,
                     axis: axes.Axes,
                     graph: nx.Graph,
                     attribute_name: str = "Pearson"):
    """draws graph edges from node to node, colored by weight

    Parameters
    ----------
    figure: matplotlib.figure.Figure
        a matplotlib Figure
    axis: matplotlib.axes.Axes
        a matplotlib Axes, part of Figure
    graph: nx.Graph
        a networkx graph, assumed to have edges formed like
        graph.add_edge((0, 1), (0, 2), weight=1.234)
    attibute_name: str
        which edge attribute to plot

    Notes
    -----
    modifes figure and axis in-place

    """
    weights = np.array([i[attribute_name] for i in graph.edges.values()])
    # graph is (row, col), transpose to get (x, y)
    segments = []
    for edge in graph.edges:
        segments.append([edge[0][::-1], edge[1][::-1]])
    line_coll = LineCollection(segments, linestyle='solid',
                               cmap="plasma", linewidths=0.3)
    line_coll.set_array(weights)
    vals = np.concatenate(line_coll.get_segments())
    mnvals = vals.min(axis=0)
    mxvals = vals.max(axis=0)
    ppvals = vals.ptp(axis=0)
    buffx = 0.02 * ppvals[0]
    buffy = 0.02 * ppvals[1]
    line_coll.set_linewidth(0.3 * 512 / ppvals[0])
    axis.add_collection(line_coll)
    axis.set_xlim(mnvals[0] - buffx, mxvals[0] + buffx)
    axis.set_ylim(mnvals[1] - buffy, mxvals[1] + buffy)
    # invert yaxis for image-like orientation
    axis.invert_yaxis()
    axis.set_aspect("equal")
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(line_coll, ax=axis, cax=cax)
    axis.set_title(attribute_name)


def create_roi_plot(plot_path: pathlib.Path,
                    img_data: np.ndarray,
                    roi_list: List[ExtractROI]) -> None:
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
    fig = figure.Figure(figsize=(40, 20))
    axes = [fig.add_subplot(1, 2, i) for i in [1, 2]]
    axes[0].imshow(img_data)
    axes[1].imshow(img_data)

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
    axes[1].imshow(bdry_pixels, cmap='autumn', alpha=1.0)
    fig.tight_layout()
    fig.savefig(plot_path)
    return None
