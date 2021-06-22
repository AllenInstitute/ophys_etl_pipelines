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
def graph_path(graph, tmpdir):
    gpath = tmpdir / "graph.pkl"
    nx.write_gpickle(graph, str(gpath))
    yield gpath


def test_draw_graph_edges(graph):
    fig = figure.Figure()
    axis = fig.add_subplot(111)
    graph_plotting.draw_graph_edges(fig, axis, graph, attribute_name="weight")


def test_draw_graph_image(graph):
    fig = figure.Figure()
    axis = fig.add_subplot(111)
    graph_plotting.draw_graph_image(fig, axis, graph, attribute_name="weight")
