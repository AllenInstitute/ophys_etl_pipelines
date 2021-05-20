import networkx as nx
import numpy as np
from typing import Union, List, Optional, Tuple, Dict


def bin_graph(graph: nx.Graph,
              bins: Union[int, List[Union[int, float]], str],
              sort_type: str = "node_count",
              attribute_name: Optional[str] = None,
              right: bool = False,
              max_inclusive: bool = True,
              ) -> Tuple[Dict[int, nx.Graph], np.ndarray]:
    """sorts the connected components of a graph by specified
    sort_type metric and returns binned combined graphs

    Parameters
    ----------
    graph: nx.Graph
        the input graph. Presumably this has multiple connected components,
        i.e. it is the result of segmentation.
    sort_type: str
        numerical metric for sorting graph connected components
        'node_count': bin components by number of nodes
        'degree': bin components by average degree
        'edge_attribute': bin components by average edge attribute
        'node_attribute': bin components by average node attribute
    attribute_name: str
        if 'sort_type' is 'edge_attribute' or 'node_attribute' this is the
        attribute name from which to extract the metric
    bins: int or list of scalars or str
        passed as 'bins' to np.histogram_bin_edges()
    right: bool
        passed as 'right' to np.digitize
    max_inclusive: bool
        if True, modifies returned values from np.digitize to make
        sure the final bin is inclusive of the endpoint


    Returns
    -------
    binned_graphs: dict
        keys are indices, as returned by np.digitize()
        values are the graphs
    bin_edges: array of dtype float
        the bin edges as returned by np.histogram_bin_edges()

    """
    subgraphs = [graph.subgraph(i) for i in nx.connected_components(graph)]

    # extract specified metric
    if sort_type == "node_count":
        values = np.array([len(i) for i in subgraphs])
    elif sort_type == "degree":
        values = [sum([i[1] for i in subgraph.degree]) / len(subgraph)
                  for subgraph in subgraphs]
    elif sort_type == "edge_attribute":
        values = [np.array(list(
                      nx.get_edge_attributes(
                          i, attribute_name).values())).mean()
                  for i in subgraphs]
    elif sort_type == "node_attribute":
        values = [np.array(list(
                      nx.get_node_attributes(
                          i, attribute_name).values())).mean()
                  for i in subgraphs]

    # set up the bins
    bin_edges = np.histogram_bin_edges(values, bins)
    if max_inclusive:
        bmax = bin_edges[-1]
        bin_edges[-1] += np.diff(values).mean() * 1e-5
    bin_indices = np.digitize(values, bin_edges)

    # collect nodes in bins
    node_lists = dict()
    for bin_index, subgraph in zip(bin_indices, subgraphs):
        if bin_index not in node_lists:
            node_lists[bin_index] = []
        node_lists[bin_index].extend(list(subgraph.nodes))

    binned_graphs = {k: graph.subgraph(v) for k, v in node_lists.items()}

    if max_inclusive:
        bin_edges[-1] = bmax

    return binned_graphs, bin_edges
