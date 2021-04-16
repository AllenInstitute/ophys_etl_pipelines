import pytest
import networkx as nx
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import ophys_etl.qc.video.correlation_graph_plot as cgp  # noqa: E402


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
    fig, axis = plt.subplots(1, 1, num=1, clear=True)
    cgp.draw_graph_edges(fig, axis, graph)


def test_correlation_graph_plot(graph_path):
    plot_path = Path(graph_path).parent / "plot.png"
    args = {
            "graph_input": str(graph_path),
            "plot_output": str(plot_path)}
    cg = cgp.CorrelationGraphPlot(input_data=args, args=[])
    cg.run()
    assert plot_path.exists()
