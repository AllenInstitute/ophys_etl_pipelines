import h5py
import networkx as nx
import numpy as np
from pathlib import Path
from scipy.spatial.distance import cdist


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
