import pytest
import tempfile
import pathlib
import numpy as np
import networkx as nx

from ophys_etl.modules.segmentation.graph_utils import creation, conversion


@pytest.fixture
def graph_fixture(request):
    attr_name = request.param.get('attribute_name')
    g = creation.create_graph(0, 99, 0, 99)
    edge_dict = {i: {attr_name: 42} for i in g.edges()}
    nx.set_edge_attributes(G=g, values=edge_dict)
    return g, attr_name


@pytest.mark.parametrize(
        "graph_fixture",
        [
            {"attribute_name": "some string"}
            ],
        indirect=["graph_fixture"])
def test_graph_to_img(graph_fixture, tmpdir):
    graph, name = graph_fixture
    img = conversion.graph_to_img(graph, attribute_name=name)
    assert isinstance(img, np.ndarray)
    assert img.shape == (100, 100)

    # test reading from file
    graph_path = tempfile.mkstemp(dir=tmpdir, suffix='.pkl')[1]
    graph_path = pathlib.Path(graph_path)
    nx.write_gpickle(graph, graph_path)
    assert graph_path.is_file()
    img = conversion.graph_to_img(graph_path, attribute_name=name)
    assert isinstance(img, np.ndarray)
    assert img.shape == (100, 100)
