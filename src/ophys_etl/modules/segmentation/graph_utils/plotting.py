import networkx as nx
import numpy as np
from typing import List, Union
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
    # graph is (row, col), transpose to get (x, y)
    edges = nx.get_edge_attributes(graph, name=attribute_name)
    segments = [np.array([edge[0][::-1], edge[1][::-1]]) for edge in edges]
    weights = np.array(list(edges.values()))
    line_coll = LineCollection(segments, linestyle='solid',
                               cmap="plasma", linewidths=0.3)
    line_coll.set_array(weights)
    vals = np.array(graph.nodes)
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


def graph_to_img(graph: Union[pathlib.Path, nx.Graph],
                 attribute: str = 'filtered_hnc_Gaussian') -> np.ndarray:
    """
    Convert a graph into a np.ndarray image

    Parameters
    ----------
    graph: Union[pathlib.Path, nx.Graph]
        Either a networkx.Graph or the path to a pickle file
        containing the graph

    attribute: str
        Name of the attribute used to create the image
        (default = 'filtered_hnc_Gaussian')

    Returns
    -------
    np.ndarray
        An image in which the value of each pixel is the
        sum of the edge weights connected to that node in
        the graph.
    """
    if isinstance(graph, pathlib.Path):
        graph = nx.read_gpickle(graph)
    else:
        if not isinstance(graph, nx.Graph):
            msg = "graph must be either a pathlib.Path or "
            msg += f"a networkx.Graph. You gave {type(graph)}"
            raise RuntimeError(msg)

    node_coords = np.array(graph.nodes).T
    row_max = node_coords[0].max()
    col_max = node_coords[1].max()
    img = np.zeros((row_max+1, col_max+1), dtype=float)
    for node in graph.nodes:
        vals = [graph[node][i][attribute] for i in graph.neighbors(node)]
        img[node[0], node[1]] = np.sum(vals)
    return img


def draw_graph_img(figure: figure.Figure,
                   axis: axes.Axes,
                   graph: nx.Graph,
                   attribute_name: str = "Pearson"):
    """draws graph as an image where every pixel's intensity is the
    sum of the edge weights connected to that pixel

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
    img = graph_to_img(graph,
                       attribute=attribute_name)
    shape = img.shape

    img = axis.imshow(img, cmap='plasma')
    axis.set_aspect("equal")
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(img, ax=axis, cax=cax)

    buffx = 0.02 * shape[1]
    buffy = 0.02 * shape[0]

    axis.set_xlim(-buffx, shape[1] + buffx)
    axis.set_ylim(-buffy, shape[0] + buffy)

    axis.set_title(attribute_name)
