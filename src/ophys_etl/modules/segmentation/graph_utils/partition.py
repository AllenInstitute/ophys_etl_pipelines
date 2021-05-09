import networkx as nx
import numpy as np
from typing import List
from sklearn.cluster import KMeans


def partition_graph_by_edges(graph: nx.Graph,
                             n_groups: int) -> List[nx.Graph]:
    """splits a graph into a list of subgraphs by k-means clustering
    of the edge centroids. The subgraphs can be recombined by
    nx.compose_all() with no loss of edges or nodes.

    Parameters
    ----------
    graph: nx.Graph
        the input graph. This routine assumes the nodes are a continuous set
        of (ri, ci) tuples that densely cover a space. i.e. the (row, col)
        coordinates of pixels in an image.
    n_groups: int
        the number of partitions

    Returns
    -------
    subgraphs: List[nx.Graph]
        the list of subgraphs

    Notes
    -----
    This function is applicable for partitioning for parallel processing
    where some edge attribute is calculated from the nodes of that edge. I.e.
    the subgraphs lose global information beyond their boundaries.

    """
    if n_groups == 1:
        return [graph]

    edges = list(graph.edges)
    centroids = [np.array(edge).mean(axis=0) for edge in edges]
    kmeans = KMeans(n_clusters=n_groups, random_state=0).fit(centroids)

    subgraphs = []
    for label in np.unique(kmeans.labels_):
        sub_edges = [e
                     for i, e in enumerate(edges)
                     if kmeans.labels_[i] == label]
        subgraphs.append(nx.Graph(graph.edge_subgraph(sub_edges)))
    return subgraphs
