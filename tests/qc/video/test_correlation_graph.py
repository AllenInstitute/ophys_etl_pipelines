import pytest
import h5py
import numpy as np
import networkx as nx
from pathlib import Path

import matplotlib as mpl
mpl.use('Agg')

import ophys_etl.qc.video.correlation_graph as cg  # noqa: E402


@pytest.fixture
def video_shape():
    return (20, 40, 40)


@pytest.fixture
def video_path(tmpdir, video_shape):
    vpath = Path(tmpdir / "video.h5")
    data = np.random.randint(0, 2**15, size=video_shape, dtype='uint16')
    with h5py.File(vpath, "w") as f:
        f.create_dataset("data", data=data)
    yield vpath


@pytest.mark.parametrize(
        "n, n_segments, expected",
        [
            (100, 1, [[0, 100]]),
            (100, 2, [[0, 50], [49, 100]]),
            (100, 3, [[0, 34], [33, 67], [66, 100]])
            ])
def test_index_split(n, n_segments, expected):
    indices = cg.index_split(n, n_segments)
    for i, e in zip(indices, expected):
        assert i == e
    # check that the first index is overlapping by at least 1 index
    for i in range(len(indices))[1:]:
        assert indices[i][0] <= (indices[i][1] - 1)


def test_weight_calculation(video_path, video_shape):
    graph = cg.weight_calculation(
            video_path,
            [0, video_shape[1]],
            [0, video_shape[2]])
    assert isinstance(graph, nx.Graph)
    # how many edges for 8 NN
    nedges = (video_shape[1] - 1) * video_shape[2]
    nedges += (video_shape[2] - 1) * video_shape[1]
    nedges += 2 * (video_shape[1] - 1) * (video_shape[2] - 1)
    assert len(graph.edges) == nedges


@pytest.mark.parametrize("n_segments", [1, 2])
def test_correlation_graph(video_path, n_segments):
    graph_path = video_path.parent / "graph.pkl"
    plot_path = video_path.parent / "graph_plot.png"
    args = {
            "video_path": str(video_path),
            "n_segments": n_segments,
            "graph_output": str(graph_path),
            "plot_output": str(plot_path)
            }
    cgraph = cg.CorrelationGraph(input_data=args, args=[])
    cgraph.run()
    assert graph_path.exists()
    assert plot_path.exists()
