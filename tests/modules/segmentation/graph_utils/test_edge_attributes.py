import pytest
import h5py
import numpy as np
import networkx as nx
from pathlib import Path

from ophys_etl.modules.segmentation.graph_utils import (
        creation, edge_attributes)


@pytest.fixture
def random_video_path(tmpdir, request):
    # random values, used to smoke test functions
    vpath = Path(tmpdir / "video.h5")
    data = np.random.randint(0, 2**15,
                             size=request.param["video_shape"],
                             dtype='uint16')
    with h5py.File(vpath, "w") as f:
        f.create_dataset("data", data=data)
    yield vpath


@pytest.fixture
def correlated_video_path(tmpdir, request):
    """simple 4-pixel video where
    set1 = (0, 0) and (1, 1) are fully correlated
    set2 = (0, 1) and (1, 0) are fully correlated
    and set1 is fully anti-correlated with set2
    """
    # specific values, used to test function numerics
    vpath = Path(tmpdir / "video.h5")
    t = np.arange(request.param["nframes"])
    # sinusoidal time series in range [0, 1]
    value = 0.5 * (1.0 + np.sin(2.0 * np.pi * t / 10.))
    data = np.moveaxis(np.array([[value, -value],
                                 [-value, value]]), 2, 0)
    with h5py.File(vpath, "w") as f:
        f.create_dataset("data", data=data)
    yield vpath


@pytest.mark.parametrize(
        "correlated_video_path",
        [
            {"nframes": 100}
        ], indirect=["correlated_video_path"])
def test_add_pearson(correlated_video_path):
    """tests that a simple 4 pixel video results in the expected
    correlation values.
    """
    with h5py.File(correlated_video_path, "r") as f:
        nrows, ncols = f["data"].shape[1:]
    g = creation.create_graph(0, nrows - 1, 0, ncols - 1)
    name = "name"
    g2 = edge_attributes.add_pearson_edge_attributes(
            graph=g,
            video_path=correlated_video_path,
            attribute_name=name)
    for node1, node2, value in g2.edges(data=True):
        distance = np.linalg.norm(np.array(node1) - np.array(node2))
        if np.isclose(distance, 1.0):
            assert np.isclose(value[name], -1.0)
        elif np.isclose(distance, np.sqrt(2)):
            assert np.isclose(value[name], 1.0)


@pytest.mark.parametrize("filter_fraction", [0.1, 0.4, 0.9])
@pytest.mark.parametrize(
        "correlated_video_path",
        [
            {"nframes": 100}
        ], indirect=["correlated_video_path"])
def test_add_filtered_pearson(correlated_video_path, filter_fraction):
    """tests that a simple 4 pixel video results in the expected
    correlation values. Given the perfect nature of the test video
    the filter fraction has no effect.
    """
    with h5py.File(correlated_video_path, "r") as f:
        nrows, ncols = f["data"].shape[1:]
    g = creation.create_graph(0, nrows - 1, 0, ncols - 1)
    name = "name"
    g2 = edge_attributes.add_filtered_pearson_edge_attributes(
            graph=g,
            video_path=correlated_video_path,
            attribute_name=name,
            filter_fraction=filter_fraction)
    for node1, node2, value in g2.edges(data=True):
        distance = np.linalg.norm(np.array(node1) - np.array(node2))
        if np.isclose(distance, 1.0):
            assert np.isclose(value[name], -1.0)
        elif np.isclose(distance, np.sqrt(2)):
            assert np.isclose(value[name], 1.0)


@pytest.mark.parametrize("filter_fraction", [0.5, 1.0])
@pytest.mark.parametrize(
        "correlated_video_path",
        [
            {"nframes": 100}
        ], indirect=["correlated_video_path"])
def test_add_hnc_gaussian(correlated_video_path, filter_fraction):
    """tests that a simple 4 pixel video results in the expected
    correlation values. Given the perfect nature of the test video
    the filter fraction has no effect.
    """
    with h5py.File(correlated_video_path, "r") as f:
        nrows, ncols = f["data"].shape[1:]
    g = creation.create_graph(0, nrows - 1, 0, ncols - 1)
    name = "name"
    g2 = edge_attributes.add_hnc_gaussian_metric(
            graph=g,
            video_path=correlated_video_path,
            attribute_name=name,
            filter_fraction=filter_fraction,
            full_neighborhood=True,
            neighborhood_radius=2)
    for node1, node2, value in g2.edges(data=True):
        distance = np.linalg.norm(np.array(node1) - np.array(node2))
        if np.isclose(distance, 1.0):
            # NOTE: this is different than for Pearson measures
            assert np.isclose(value[name], 0.0, atol=1e-6)
        elif np.isclose(distance, np.sqrt(2)):
            assert np.isclose(value[name], 1.0)


@pytest.mark.parametrize(
        "random_video_path",
        [
            {"video_shape": (20, 40, 40)},
            {"video_shape": (10, 8, 40)}
        ], indirect=["random_video_path"])
@pytest.mark.parametrize(
        "edge_calc_callable, edge_calc_kwargs",
        [
            (edge_attributes.add_pearson_edge_attributes,
             {"attribute_name": "Pearson"}),
            (edge_attributes.add_filtered_pearson_edge_attributes,
             {"attribute_name": "f_Pearson",
              "filter_fraction": 0.2}),
            (edge_attributes.add_hnc_gaussian_metric,
             {"attribute_name": "hnc_Gauss",
              "neighborhood_radius": 3,
              "full_neighborhood": False,
              "filter_fraction": None}),
            (edge_attributes.add_hnc_gaussian_metric,
             {"attribute_name": "hnc_Gauss",
              "neighborhood_radius": 3,
              "full_neighborhood": False,
              "filter_fraction": 0.2}),
            (edge_attributes.add_hnc_gaussian_metric,
             {"attribute_name": "hnc_Gauss",
              "neighborhood_radius": 3,
              "full_neighborhood": True,
              "filter_fraction": 0.2}),
            ])
def test_add_edge_attributes_smoke(random_video_path,
                                   edge_calc_callable,
                                   edge_calc_kwargs):
    with h5py.File(random_video_path, "r") as f:
        nrow, ncol = f["data"].shape[1:]
    graph = creation.create_graph(row_min=0, row_max=(nrow - 1),
                                  col_min=0, col_max=(ncol - 1))
    # add node attributes
    na = {n: {"node_attr": i} for i, n in enumerate(graph.nodes)}
    nx.set_node_attributes(graph, na)
    # add edge attributes
    ea = {e: {"edge_attr": i} for i, e in enumerate(graph.edges)}
    nx.set_edge_attributes(G=graph, values=ea)

    # original graph has only "edge_attr"
    for n1, n2, attr in graph.edges(data=True):
        keys = list(attr.keys())
        assert len(keys) == 1
        assert "edge_attr" in keys

    # add Pearson edge attribute by routine
    graph_with_edges = edge_calc_callable(graph=graph,
                                          video_path=random_video_path,
                                          **edge_calc_kwargs)

    # new graph has new edge attribute "Pearson"
    for n1, n2, attr in graph_with_edges.edges(data=True):
        keys = list(attr.keys())
        assert len(keys) == 2
        assert edge_calc_kwargs["attribute_name"] in keys
        assert "edge_attr" in keys

    # make sure original node attributes came along
    assert "node_attr" in list(graph_with_edges.nodes(data=True))[0][1]


@pytest.mark.parametrize(
        "random_video_path",
        [
            {"video_shape": (20, 40, 40)},
        ], indirect=["random_video_path"])
def test_normalize_graph(random_video_path):
    with h5py.File(random_video_path, "r") as f:
        nrow, ncol = f["data"].shape[1:]
    graph = creation.create_graph(row_min=0, row_max=(nrow - 1),
                                  col_min=0, col_max=(ncol - 1))
    graph = edge_attributes.add_pearson_edge_attributes(
            graph=graph,
            video_path=random_video_path)
    for n1, n2, attr in graph.edges(data=True):
        keys = list(attr.keys())
        assert "Pearson" in keys
        assert "Pearson_normalized" not in keys
    graph = edge_attributes.normalize_graph(graph, attribute_name="Pearson")
    for n1, n2, attr in graph.edges(data=True):
        keys = list(attr.keys())
        assert "Pearson" in keys
        assert "Pearson_normalized" in keys


@pytest.mark.parametrize(
        "random_video_path",
        [
            {"video_shape": (20, 40, 40)},
        ], indirect=["random_video_path"])
def test_filtered_pearson_edges(random_video_path):
    with h5py.File(random_video_path, "r") as f:
        nrow, ncol = f["data"].shape[1:]
    graph = creation.create_graph(row_min=0, row_max=(nrow - 1),
                                  col_min=0, col_max=(ncol - 1))
    graph = edge_attributes.add_filtered_pearson_edge_attributes(
            filter_fraction=0.2,
            graph=graph,
            video_path=random_video_path)
    for n1, n2, attr in graph.edges(data=True):
        keys = list(attr.keys())
        assert "filtered_Pearson" in keys
        assert "Pearson" not in keys
        assert "Pearson_normalized" not in keys


@pytest.mark.parametrize(
        "coordinates, radius, nrow, ncol, expected",
        [
            (
                # negative pixel coordinates are not returned
                (0.0, 0.0), 1, 2, 2,
                [[0, 0], [0, 1], [1, 0]]),
            (
                # pixel coordinates beyond bounds are not returned
                (1.0, 1.0), 1, 2, 2,
                [[1, 1], [0, 1], [1, 0]]),
            (
                # coordinates and radius can be floats
                (1.5, 1.5), 1.3, 5, 5,
                [[1, 1], [1, 2], [2, 1], [2, 2]]),
            (
                # coordinates and radius can be floats
                (1.5, 1.5), 1.6, 5, 5,
                [[1, 1], [1, 2], [2, 1], [2, 2],
                 [0, 1], [0, 2], [1, 0], [2, 0],
                 [3, 1], [3, 2], [1, 3], [2, 3]])
            ])
def test_neighborhood_pixels(coordinates, radius, nrow, ncol, expected):
    cc = edge_attributes.neighborhood_pixels(coordinates, radius, nrow, ncol)
    set_cc = set([tuple(i) for i in cc])
    set_expected = set([tuple(i) for i in expected])
    assert set_cc == set_expected
