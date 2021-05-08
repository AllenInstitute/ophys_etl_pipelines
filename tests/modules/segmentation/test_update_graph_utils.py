import pytest
import itertools
import networkx as nx

import ophys_etl.modules.segmentation.modules.update_graph.utils as ugu
from ophys_etl.modules.segmentation.modules.create_graph.__main__ \
    import create_graph


@pytest.mark.parametrize(
        "graph, n_groups",
        [
            (create_graph(0, 1, 0, 1), 2),
            (create_graph(0, 8, 0, 8), 2),
            (create_graph(11, 34, 4, 53), 7)
        ])
def test_partition_graph_by_edges(graph, n_groups):
    # add node attributes
    na = {n: {"node_attr": i} for i, n in enumerate(graph.nodes)}
    nx.set_node_attributes(graph, na)
    # add edge attributes
    ea = {e: {"edge_attr": i} for i, e in enumerate(graph.edges)}
    nx.set_edge_attributes(graph, ea)

    subgraphs = ugu.partition_graph_by_edges(graph, n_groups)

    # check that the graph can be re-assembled completely from the subgraphs
    composed = nx.compose_all(subgraphs)
    # check the nodes and edges the same
    for a, b in itertools.permutations([graph, composed]):
        for node in a.nodes:
            assert b.has_node(node)
        for edge in a.edges:
            assert b.has_edge(*edge)

    # check the first node and edge have the named attributes
    assert "node_attr" in list(composed.nodes(data=True))[0][1]
    assert "edge_attr" in list(composed.edges(data=True))[0][2]

    # check all the nodes and edges have the correct attribute values
    assert nx.get_node_attributes(graph, "node_attr") == \
        nx.get_node_attributes(composed, "node_attr")
    for n1, n2, value in graph.edges(data=True):
        assert composed.get_edge_data(n1, n2)["edge_attr"] == \
            value["edge_attr"]
