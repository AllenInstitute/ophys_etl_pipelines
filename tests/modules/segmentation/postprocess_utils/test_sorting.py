import pytest
import networkx as nx

from ophys_etl.modules.segmentation.postprocess_utils import sorting


@pytest.fixture
def graph_to_bin():
    """a graph whose connected components are distinguishable
    by node attribute, edge attribute, node count, and degree
    """
    g = nx.Graph()

    # first connected_component
    g.add_node(1, node_attr=0)
    g.add_node(2, node_attr=0)
    g.add_node(3, node_attr=0)
    g.add_edge(1, 2, edge_attr=0)
    g.add_edge(1, 3, edge_attr=0)
    g.add_edge(2, 3, edge_attr=0)

    # second connected_component
    g.add_node(4, node_attr=1)
    g.add_node(5, node_attr=1)
    g.add_node(6, node_attr=1)
    g.add_node(7, node_attr=1)
    g.add_edge(4, 5, edge_attr=1)
    g.add_edge(4, 6, edge_attr=1)
    g.add_edge(4, 7, edge_attr=1)
    g.add_edge(5, 6, edge_attr=1)
    g.add_edge(5, 7, edge_attr=1)
    g.add_edge(6, 7, edge_attr=1)

    return g


@pytest.mark.parametrize(
        "sort_type, attribute_name",
        [
            ("node_count", None),
            ("degree", None),
            ("edge_attribute", "edge_attr"),
            ("node_attribute", "node_attr")])
def test_bin_graph(graph_to_bin, sort_type, attribute_name):
    binned_graphs, bins = sorting.bin_graph(graph_to_bin,
                                            bins=2,
                                            sort_type=sort_type,
                                            attribute_name=attribute_name)
    assert bins.size == 3
    assert len(binned_graphs) == 2
    for v in binned_graphs.values():
        assert isinstance(v, nx.Graph)
    assert len(binned_graphs[1]) == 3
    assert len(binned_graphs[1].edges) == 3
    assert len(binned_graphs[2]) == 4
    assert len(binned_graphs[2].edges) == 6
