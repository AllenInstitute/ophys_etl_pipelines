import networkx as nx
import numpy as np
from typing import List


def edge_values(graph: nx.Graph, attribute_name: str) -> np.ndarray:
    """returns an array of the values for 'attribute_name'
    from a graph

    Parameters
    ----------
    graph: nx.Graph
        the input graph
    attribute_name: str
        the name of the edge attribute

    Returns
    -------
    values: np.ndarray
        array of the values

    """
    attr = np.array([v for v in nx.get_edge_attributes(
        graph, name=attribute_name).values()])
    return attr


def seed_subgraphs_by_quantile(graph: nx.Graph,
                               attribute_name: str,
                               seed_quantile: float,
                               n_node_thresh: int,
                               ) -> List[nx.Graph]:
    """idenitfy seeds, grow out ROIs, repeat

    Parameters
    ----------
    graph: nx.Graph
        the input graph
    attribute_name: str
        the name of the edge attribute
    seed_quantile: float
        establishes threshold for finding seeds
    n_node_thresh: int
        seeds smaller than this are disregarded

    Returns
    -------
    subgraphs: List[nx.Graph]
        the identified seeds as their own graphs

    """

    values = edge_values(graph, attribute_name)
    thresh = np.quantile(values, seed_quantile)

    new_graph = nx.Graph()
    for n1, n2, edge in graph.edges(data=True):
        if edge[attribute_name] > thresh:
            new_graph.add_edge(n1, n2, **edge)
    connected = list(nx.connected_components(new_graph))
    # ignore small ones
    cc = [i for i in connected
          if len(i) > n_node_thresh]
    subgraphs = []
    for icc in cc:
        # eliminate purely linear graphs
        # maybe not a problem if we ignore or resolve
        # the rectilinear correlation distribution
        coords = np.array(list(icc))
        if (np.unique(coords[:, 0]).size == 1) | \
                (np.unique(coords[:, 1]).size == 1):
            continue
        subgraphs.append(graph.subgraph(icc))

    return subgraphs


def grow_subgraph(graph: nx.Graph,
                  subgraph: nx.Graph,
                  attribute_name: str) -> nx.Graph:
    """grow a subgraph according to some criteria

    Parameters
    ----------
    graph: nx.Graph
        the parent graph, used to supply potential new nodes
    subgraph: nx.Graph
        the starting subgraph
    attribute_name: str
        the name of the edge attribute

    Returns
    -------
    last_sub: nx.Graph
        the result of growing the subgraph

    """
    n_start = len(subgraph)
    n_nodes = 1
    last_count = 0
    last_sub = nx.Graph(subgraph)
    while n_nodes > last_count:
        last_count = len(last_sub)
        vals = edge_values(last_sub, attribute_name=attribute_name)

        if vals.size == 0:
            # this seemed to happen, how?
            return None

        # hard-coded (for now) criteria: can go dip_fraction below the
        # minimum edge (could be > or < 0)
        # if dip_fraction > 0 allows graphs to grow to less correlated nodes
        dip_fraction = 0.1
        thresh = vals.min() - (dip_fraction * vals.ptp())

        if (vals.ptp() / vals.max()) > 0.05:
            # terminates growth if the variation across the ROI gets too big
            break

        # get next layer of neighbors from this graph
        neighbors = []
        for n in last_sub:
            neighbors.extend([i for i in graph.neighbors(n)
                              if i not in last_sub])

        # cycle through candidate neighbors
        values = []
        edges = []
        for n in neighbors:
            # for each neighbor candidate, connecting edges are
            # from last_sub are:
            node_edges = []
            for sn in last_sub.nodes:
                v = graph.get_edge_data(n, sn)
                if v is not None:
                    node_edges.append((n, sn, v))
            node_edge_values = [i[2][attribute_name] for i in node_edges]
            edges.append(node_edges)
            # for this neighbor, the mean edge values connecting to last_sub
            values.append(np.mean(node_edge_values))

        # for the highest valued node, add all its edges
        ind = np.argmax(values)
        if values[ind] > thresh:
            edges = edges[ind]
            for edge in edges:
                last_sub.add_edge(edge[0],
                                  edge[1],
                                  **edge[2])

        n_nodes = len(last_sub)
        if n_nodes > 200:
            # in low-contrast regions, sometimes growth never stopped
            return last_sub

    print(f"grew subgraph from {n_start} to {n_nodes}")
    return last_sub


def iterative_detection(graph: nx.Graph,
                        attribute_name: str,
                        seed_quantile: int,
                        n_node_thresh=20) -> nx.Graph:
    """idenitfy seeds, grow out ROIs, repeat

    Parameters
    ----------
    graph: nx.Graph
        the input graph
    attribute_name: str
        the name of the edge attribute
    seed_quantile: float
        establishes threshold for finding seeds
    n_node_thresh: int
        seeds smaller than this are disregarded

    Returns
    -------
    graph: nx.Graph
        a graph consisting of only identified ROIs

    """
    collected = []
    for jj in range(5):
        subgraphs = seed_subgraphs_by_quantile(graph,
                                               attribute_name=attribute_name,
                                               seed_quantile=seed_quantile,
                                               n_node_thresh=n_node_thresh)
        if len(subgraphs) == 0:
            break
        expanded = []
        for subgraph in subgraphs:
            sub_nodes = set(subgraph.nodes) & set(graph.nodes)
            subgraph = graph.subgraph(sub_nodes)
            expanded_subgraph = grow_subgraph(graph,
                                              subgraph,
                                              attribute_name)
            if expanded_subgraph is None:
                continue
            node_list = set(graph.nodes) - set(expanded_subgraph.nodes)
            graph = graph.subgraph(node_list)
            expanded.append(expanded_subgraph)
        expanded = nx.compose_all(expanded)

        nodes = [i for i in graph if i not in expanded]
        graph = nx.Graph(graph.subgraph(nodes))
        collected.append(expanded)

    return nx.compose_all(collected)
