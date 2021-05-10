import networkx as nx
import numpy as np
from matplotlib import figure, axes
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
