import networkx as nx
import numpy as np
import multiprocessing
from pathlib import Path
from typing import List, Union, Optional

from ophys_etl.types import ExtractROI


def graph_to_roi(graph: nx.Graph, roi_id: Optional[int] = 0) -> ExtractROI:
    """LIMS style ROI from a graph

    Parameters
    ----------
    graph: nx.Graph
        an input graph, probably a smaller graph resulting from some
        segmentation and/or other processing
    roi_id: int
        idenitfier for this ROI

    Returns
    -------
    roi: ExtractROI
        the LIMS-formatted ROI

    """
    coords = np.array(list(graph.nodes))
    row0 = coords[:, 0].min()
    col0 = coords[:, 1].min()
    cptp = tuple(coords.ptp(axis=0))
    mask = np.zeros((cptp[0] + 1, cptp[1] + 1), dtype=bool)
    for row, col in coords:
        mask[row - row0, col - col0] = True
    roi = ExtractROI(
            id=roi_id,
            x=int(col0),
            y=int(row0),
            width=int(cptp[1]),
            height=int(cptp[0]),
            valid=False,
            mask=[i.tolist() for i in mask])
    return roi


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

        if len(values) == 0:
            return last_sub

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

    return last_sub


def _process_subgraphs(subgraphs,
                       graph,
                       attribute_name,
                       p_id,
                       out_dict):

    local_expanded = []
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
        local_expanded.append(expanded_subgraph)

    if len(local_expanded) == 0:
        return None

    full_graph = nx.compose_all(local_expanded)
    out_dict[p_id] = full_graph


def iterative_detection(graph: Union[nx.Graph, Path],
                        attribute_name: str,
                        seed_quantile: int,
                        n_node_thresh=20,
                        n_processes=1) -> Union[nx.Graph, Path]:
    """idenitfy seeds, grow out ROIs, repeat

    Parameters
    ----------
    graph: nx.Graph
        the input graph, or a path to such
    attribute_name: str
        the name of the edge attribute
    seed_quantile: float
        establishes threshold for finding seeds
    n_node_thresh: int
        seeds smaller than this are disregarded
    n_processes: int
        number of processors to run

    Returns
    -------
    graph: nx.Graph
        a graph consisting of only identified ROIs, or a path to such

    """
    from_path = None
    if isinstance(graph, Path):
        from_path = Path(graph)
        graph = nx.read_gpickle(graph)

    rng = np.random.RandomState(1182)

    collected = []
    for jj in range(5):
        subgraphs = seed_subgraphs_by_quantile(graph,
                                               attribute_name=attribute_name,
                                               seed_quantile=seed_quantile,
                                               n_node_thresh=n_node_thresh)
        if len(subgraphs) == 0:
            break

        rng.shuffle(subgraphs)
        many_graphs = False
        if n_processes == 1:
            out_dict = {}
            _process_subgraphs(subgraphs, graph, attribute_name, 0, out_dict)
            expanded = out_dict[0]
        else:

            slop = 4
            n_subgraphs = len(subgraphs)
            d_graph = n_subgraphs//(slop*n_processes)

            # the slop factor is so that each process
            # gets the chance to take on more than one group
            # of subgraphs, in case one group is messier
            # than another (which is likely true)
            while (slop*n_processes)*d_graph < n_subgraphs:
                d_graph += 1

            p_list = []
            mgr = multiprocessing.Manager()
            out_dict = mgr.dict()

            for i_start in range(0, n_subgraphs, d_graph):
                p = multiprocessing.Process(target=_process_subgraphs,
                                            args=(subgraphs[i_start:i_start+d_graph],
                                                  graph, attribute_name, i_start,
                                                  out_dict))
                p.start()
                p_list.append(p)
                while len(p_list) >= n_processes:
                    to_pop = []
                    for ii in range(len(p_list)-1,-1,-1):
                        if p_list[ii].exitcode is not None:
                            to_pop.append(ii)
                    for ii in to_pop:
                        p_list.pop(ii)
            for p in p_list:
                p.join()

            many_graphs = True
            expanded = []
            for ii in out_dict:
                if out_dict[ii] is not None:
                    expanded.append(out_dict[ii])

        if many_graphs:
            expanded = nx.compose_all(expanded)

        nodes = [i for i in graph if i not in expanded]
        graph = nx.Graph(graph.subgraph(nodes))
        collected.append(expanded)

    graph = nx.compose_all(collected)

    if from_path is not None:
        nx.write_gpickle(graph, from_path)
        graph = from_path

    return graph
