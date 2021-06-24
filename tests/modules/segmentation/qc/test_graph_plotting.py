import pytest
import networkx as nx
from matplotlib import figure

from ophys_etl.modules.segmentation.qc import graph_plotting


@pytest.fixture
def graph():
    g = nx.Graph()
    g.add_edge((0, 0), (0, 1), weight=1.0)
    g.add_edge((0, 0), (0, 2), weight=1.1)
    g.add_edge((1, 0), (0, 2), weight=1.2)
    return g


@pytest.fixture
def graph_with_names(request):
    edges = [[(0, 0), (0, 1)],
             [(0, 0), (0, 2)],
             [(1, 0), (0, 2)]]
    g = nx.Graph()
    for edge in edges:
        g.add_edge(*edge)
    for name in request.param.get("names"):
        values = {e: {name: 42} for e in g.edges}
        nx.set_edge_attributes(G=g, values=values)
    return g, request.param.get("names")


@pytest.fixture
def graph_path(graph, tmpdir):
    gpath = tmpdir / "graph.pkl"
    nx.write_gpickle(graph, str(gpath))
    yield gpath


@pytest.mark.parametrize(
        "graph_with_names",
        [
            {"names": []},
            {"names": ["a"]},
            {"names": ["a", "b"]}],
        indirect=["graph_with_names"])
def test_find_graph_edge_attribute_names(graph_with_names):
    graph, names = graph_with_names
    found_names = set(graph_plotting.find_graph_edge_attribute_names(graph))
    assert found_names == set(names)


def test_draw_graph_edges(graph):
    fig = figure.Figure()
    axis = fig.add_subplot(111)
    graph_plotting.draw_graph_edges(fig, axis, graph, attribute_name="weight")


def test_draw_graph_image(graph):
    fig = figure.Figure()
    axis = fig.add_subplot(111)
    graph_plotting.draw_graph_image(fig, axis, graph, attribute_name="weight")
