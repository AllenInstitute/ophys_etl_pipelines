import pytest
import networkx as nx
from pathlib import Path

from ophys_etl.modules.segmentation.modules.create_plot import GraphPlot


@pytest.fixture
def graph_path(tmpdir):
    g = nx.Graph()
    g.add_edge((0, 0), (0, 1), weight=1.0)
    g.add_edge((0, 0), (0, 2), weight=1.1)
    g.add_edge((1, 0), (0, 2), weight=1.2)
    gpath = tmpdir / "graph.pkl"
    nx.write_gpickle(g, str(gpath))
    yield gpath


@pytest.mark.parametrize("draw_edges", [True, False])
def test_GraphPlot(graph_path, draw_edges, tmpdir):
    plot_output = tmpdir / "plot.png"
    args = {
            "graph_input": str(graph_path),
            "plot_output": str(plot_output),
            "draw_edges": draw_edges}
    gp = GraphPlot(input_data=args, args=[])
    gp.run()
    assert Path(plot_output).exists()
