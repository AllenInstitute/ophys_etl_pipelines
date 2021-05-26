import pytest
import pathlib
import networkx as nx
import h5py
import tempfile
import numpy as np


@pytest.fixture
def example_graph(tmpdir):
    graph_path = pathlib.Path(tempfile.mkstemp(
                                      dir=tmpdir,
                                      prefix='graph_',
                                      suffix='.pkl')[1])

    rng = np.random.RandomState(5813)

    graph = nx.Graph()

    img = np.zeros((40, 40), dtype=int)
    img[12:16, 4:7] = 1
    img[25:32, 11:18] = 1
    img[25:27, 15:18] = 0

    for r0 in range(40):
        for dr in (-1, 1):
            r1 = r0+dr
            if r1 < 0 or r1 >= 40:
                continue
            for c0 in range(40):
                for dc in (-1, 1):
                    c1 = c0 + dc
                    if c1 < 0 or c1 >= 40:
                        continue
                    if img[r0, c0] > 0 and img[r1, c1] > 0:
                        v = np.abs(rng.normal(0.5, 0.1))
                    else:
                        v = np.abs(rng.normal(0.1, 0.05))
                    graph.add_edge((r0, c0), (r1, c1),
                                   dummy_attribute=v)

    nx.write_gpickle(graph, graph_path)
    return graph_path


@pytest.fixture
def example_video(tmpdir):
    """
    Create an example video with a non-random trace in the
    ROI footprint define in example_graph
    """
    video_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir,
                                               prefix='video_',
                                               suffix='.h5')[1])
    rng = np.random.RandomState(1245523)
    data = rng.random_sample((100, 40, 40))
    for tt in range(30, 40, 1):
        data[tt, 12:16, 4:7] += (30-tt)
    tt = np.arange(60, 88, dtype=int)
    ss = 5.0*np.exp(((tt-75)/5)**2)
    for ir in range(12, 16):
        for ic in range(4, 7):
            data[60:88, ir, ic] += ss

    mask = np.zeros((40, 40), dtype=int)
    mask[25:32, 11:19] = True
    mask[15:27, 15:18] = False
    tt = np.arange(100, dtype=int)
    ss = np.sin(2.0*np.pi*tt/25.0)
    ss = np.where(ss > 0.0, ss, 0.0)
    for ir in range(40):
        for ic in range(40):
            if mask[ir, ic]:
                data[:, ir, ic] += ss

    with h5py.File(video_path, 'w') as out_file:
        out_file.create_dataset('data', data=data)
    return video_path


@pytest.fixture
def blank_graph(tmpdir):
    graph_path = pathlib.Path(tempfile.mkstemp(
                                      dir=tmpdir,
                                      prefix='graph_',
                                      suffix='.pkl')[1])

    graph = nx.Graph()

    img = np.zeros((40, 40), dtype=int)
    img[12:16, 4:7] = 1
    img[25:32, 11:18] = 1
    img[25:27, 15:18] = 0

    for r0 in range(40):
        for dr in (-1, 1):
            r1 = r0+dr
            if r1 < 0 or r1 >= 40:
                continue
            for c0 in range(40):
                for dc in (-1, 1):
                    c1 = c0 + dc
                    if c1 < 0 or c1 >= 40:
                        continue
                    graph.add_edge((r0, c0), (r1, c1),
                                   dummy_attribute=0.0)

    nx.write_gpickle(graph, graph_path)
    return graph_path


@pytest.fixture
def blank_video(tmpdir):
    """
    Create an example video with a non-random trace in the
    ROI footprint define in example_graph
    """
    video_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir,
                                               prefix='video_',
                                               suffix='.h5')[1])
    data = np.zeros((100, 40, 40), dtype=int)

    with h5py.File(video_path, 'w') as out_file:
        out_file.create_dataset('data', data=data)
    return video_path


@pytest.fixture
def example_img():
    """
    A numpy array with known peaks in random data
    """
    img_shape = (20, 20)
    rng = np.random.RandomState(213455)

    img = rng.randint(0, 2, size=img_shape)

    img[2, 3] = 12
    img[11, 12] = 11
    img[10, 11] = 10  # too close to be detected
    return img
