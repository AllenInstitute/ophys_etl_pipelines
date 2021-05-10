import h5py
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Optional
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter


def normalize_graph(graph: nx.Graph,
                    attribute_name: str,
                    sigma: float = 30.0,
                    new_attribute_name: Optional[str] = None) -> nx.Graph:
    """normalizes edge weights by a local Gaussian filter
    of size sigma.

    Parameters
    ----------
    graph: nx.Graph
        a graph with edge weights
    sigma: float
        passed to scipy.ndimage.gaussian_filter as 'sigma'
    attribute_name: str
        the name of the edge attribute to normalize
    new_attribute_name: str
        the name of the new edge attribute. If 'None' will be
        <attribute_name>_normalized

    Returns
    -------
    graph: nx.Graph
        graph with the new edge attribute

    """
    if new_attribute_name is None:
        new_attribute_name = attribute_name + "_normalized"

    # create an image that is the average edge attribute per node
    coords = np.array(list(graph.nodes))
    shape = tuple(coords.max(axis=0) + 1)
    avg_attr = np.zeros(shape, dtype='float')
    for node in graph.nodes:
        n = len(list(graph.neighbors(node)))
        avg_attr[node[0], node[1]] = \
            graph.degree(node, weight=attribute_name) / n

    # Gaussian filter the average image
    avg_attr = gaussian_filter(avg_attr, sigma, mode='nearest')
    edge_values = dict()
    for i, v in graph.edges.items():
        local = 0.5 * (avg_attr[i[0][0], i[0][1]] + avg_attr[i[1][0], i[1][0]])
        edge_values[i] = {new_attribute_name: v[attribute_name] / local}

    nx.set_edge_attributes(graph, edge_values)
    return graph


def add_pearson_edge_attributes(graph: nx.Graph,
                                video_path: Path,
                                attribute_name: str = "Pearson") -> nx.Graph:
    """adds an attribute to each edge which is the Pearson correlation
    coefficient between the traces of the two pixels (nodes) associated
    with that edge.

    Parameters
    ----------
    graph: nx.Graph
        a graph with nodes like (row, col) and edges connecting them
    video_path: Path
        path to an hdf5 video file, assumed to have a dataset "data"
        nframes x nrow x ncol
    attribute_name: str
        name set on each edge for this calculated value

    Returns
    -------
    new_graph: nx.Graph
        an undirected networkx graph, with attribute added to edges

    """
    new_graph = nx.Graph()
    # copies over node attributes
    new_graph.add_nodes_from(graph.nodes(data=True))

    # load the section of data that encompasses this graph
    rows, cols = np.array(graph.nodes).T
    with h5py.File(video_path, "r") as f:
        data = f["data"][:,
                         rows.min(): (rows.max() + 1),
                         cols.min(): (cols.max() + 1)]

    offset = np.array([rows.min(), cols.min()])
    for node1 in graph:
        neighbors = set(list(graph.neighbors(node1)))
        new_neighbors = set(list(new_graph.neighbors(node1)))
        neighbors = list(neighbors - new_neighbors)
        if len(neighbors) == 0:
            continue
        nrow, ncol = np.array(node1) - offset
        irows, icols = (np.array(neighbors) - offset).T
        weights = 1.0 - cdist([data[:, nrow, ncol]],
                              [data[:, r, c] for r, c in zip(irows, icols)],
                              metric="correlation")[0]
        for node2, weight in zip(neighbors, weights):
            attr = graph.get_edge_data(node1, node2)
            attr.update({attribute_name: weight})
            new_graph.add_edge(node1, node2, **attr)

    return new_graph
