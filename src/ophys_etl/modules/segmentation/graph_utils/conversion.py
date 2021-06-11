import networkx as nx
import numpy as np
from typing import Union
import pathlib


def graph_to_img(graph: Union[pathlib.Path, nx.Graph],
                 attribute_name: str = 'filtered_hnc_Gaussian') -> np.ndarray:
    """
    Convert a graph into a np.ndarray image

    Parameters
    ----------
    graph: Union[pathlib.Path, nx.Graph]
        Either a networkx.Graph or the path to a pickle file
        containing the graph

    attribute_name: str
        Name of the attribute used to create the image
        (default = 'filtered_hnc_Gaussian')

    Returns
    -------
    np.ndarray
        An image in which the value of each pixel is the
        mean of the edge weights connected to that node in
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
        vals = [graph[node][i][attribute_name]
                for i in graph.neighbors(node)]
        img[node[0], node[1]] = np.mean(vals)
    return img
