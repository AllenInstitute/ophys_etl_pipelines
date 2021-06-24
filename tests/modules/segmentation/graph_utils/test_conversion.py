import pytest
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
def test_graph_to_img(graph_fixture):
    graph, name = graph_fixture
    img = conversion.graph_to_img(graph, attribute_name=name)
    assert isinstance(img, np.ndarray)
    assert img.shape == (100, 100)
