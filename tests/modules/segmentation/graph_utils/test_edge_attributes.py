import pytest
import h5py
import numpy as np
import networkx as nx
from pathlib import Path

from ophys_etl.modules.segmentation.graph_utils import (
        creation, edge_attributes)


@pytest.fixture
def video_path(tmpdir, request):
    vpath = Path(tmpdir / "video.h5")
    data = np.random.randint(0, 2**15,
                             size=request.param["video_shape"],
                             dtype='uint16')
    with h5py.File(vpath, "w") as f:
        f.create_dataset("data", data=data)
    yield vpath


@pytest.mark.parametrize(
        "video_path",
        [
            {"video_shape": (20, 40, 40)},
            {"video_shape": (10, 8, 40)}
        ], indirect=["video_path"])
def test_add_pearson_edge_attributes(video_path):
    with h5py.File(video_path, "r") as f:
        nrow, ncol = f["data"].shape[1:]
    graph = creation.create_graph(row_min=0, row_max=(nrow - 1),
                                  col_min=0, col_max=(ncol - 1))
    # add node attributes
    na = {n: {"node_attr": i} for i, n in enumerate(graph.nodes)}
    nx.set_node_attributes(graph, na)
    # add edge attributes
    ea = {e: {"edge_attr": i} for i, e in enumerate(graph.edges)}
    nx.set_edge_attributes(graph, ea)

    # original graph has only "edge_attr"
    for n1, n2, attr in graph.edges(data=True):
        keys = list(attr.keys())
        assert len(keys) == 1
        assert "edge_attr" in keys

    # add Pearson edge attribute by routine
    graph_with_edges = edge_attributes.add_pearson_edge_attributes(
            graph, video_path)

    # new graph has new edge attribute "Pearson"
    for n1, n2, attr in graph_with_edges.edges(data=True):
        keys = list(attr.keys())
        assert len(keys) == 2
        assert "Pearson" in keys
        assert "edge_attr" in keys

    # make sure original node attributes came along
    assert "node_attr" in list(graph_with_edges.nodes(data=True))[0][1]


@pytest.mark.parametrize(
        "video_path",
        [
            {"video_shape": (20, 40, 40)},
        ], indirect=["video_path"])
def test_normalize_graph(video_path):
    with h5py.File(video_path, "r") as f:
        nrow, ncol = f["data"].shape[1:]
    graph = creation.create_graph(row_min=0, row_max=(nrow - 1),
                                  col_min=0, col_max=(ncol - 1))
    graph = edge_attributes.add_pearson_edge_attributes(graph, video_path)
    for n1, n2, attr in graph.edges(data=True):
        keys = list(attr.keys())
        assert "Pearson" in keys
        assert "Pearson_normalized" not in keys
    graph = edge_attributes.normalize_graph(graph, attribute_name="Pearson")
    for n1, n2, attr in graph.edges(data=True):
        keys = list(attr.keys())
        assert "Pearson" in keys
        assert "Pearson_normalized" in keys
