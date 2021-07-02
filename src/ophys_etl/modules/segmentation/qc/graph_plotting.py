import networkx as nx
import numpy as np
from typing import List, Optional
from matplotlib import figure, axes
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ophys_etl.modules.segmentation.graph_utils.conversion import \
    graph_to_img


def find_graph_edge_attribute_names(graph: nx.Graph) -> List[str]:
    """return the edge attribute names from the graph

    Parameters
    ----------
    graph: nx.Graph
        the graph

    Returns
    -------
    names: List[str]
        the edge_attribute names

    """
    names = set()
    for _, _, attr in graph.edges(data=True):
        for k in attr:
            names.add(k)
    names = list(names)
    return names


def draw_graph_edges(figure: figure.Figure,
                     axis: axes.Axes,
                     graph: nx.Graph,
                     attribute_name: Optional[str] = None,
                     colorbar: bool = True):
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
        which edge attribute to plot. If None, will try to find the name

    Notes
    -----
    modifes figure and axis in-place

    """
    if attribute_name is None:
        names = find_graph_edge_attribute_names(graph)
        if len(names) != 1:
            raise ValueError("'attribute_name' was not specified, but when "
                             "searching for one and only one name, found "
                             f"{len(names)}: {names}. Specify "
                             "'attribute_name'")
        attribute_name = names[0]

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
    buffx = 0.02 * ppvals[1]
    buffy = 0.02 * ppvals[0]

    line_coll.set_linewidth(0.3 * 512 / ppvals[0])
    axis.add_collection(line_coll)
    axis.set_xlim(mnvals[1] - buffx, mxvals[1] + buffx)
    axis.set_ylim(mnvals[0] - buffy, mxvals[0] + buffy)

    # invert yaxis for image-like orientation
    axis.invert_yaxis()
    axis.set_aspect("equal")
    if colorbar:
        divider = make_axes_locatable(axis)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        figure.colorbar(line_coll, ax=axis, cax=cax)
    axis.set_title(attribute_name)


def draw_graph_image(figure: figure.Figure,
                     axis: axes.Axes,
                     graph: nx.Graph,
                     attribute_name: Optional[str] = None):
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
    if attribute_name is None:
        names = find_graph_edge_attribute_names(graph)
        if len(names) != 1:
            raise ValueError("'attribute_name' was not specified, but when "
                             "searching for one and only one name, found "
                             f"{len(names)}: {names}. Specify "
                             "'attribute_name'")
        attribute_name = names[0]

    img = graph_to_img(graph,
                       attribute_name=attribute_name)
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
