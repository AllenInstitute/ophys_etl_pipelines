import pytest

import pathlib
import h5py
import numpy as np
import networkx as nx
import tempfile
import json

from ophys_etl.types import ExtractROI

from ophys_etl.modules.segmentation.graph_utils.feature_vector_segmentation import (
    convert_to_lims_roi,
    graph_to_img,
    find_peaks,
    calculate_pearson_feature_vectors,
    FeatureVectorSegmenter)


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


@pytest.mark.parametrize(
    "origin,mask,expected",
    [
     ((14, 22), np.array([[False, False, False, False, False],
                          [False, True, False, False, False],
                          [False, False, True, False, False],
                          [False, False, False, False, False],
                          [False, False, False, False, False]]),
      ExtractROI(
          id=0,
          x=23,
          y=15,
          width=2,
          height=2,
          mask=[[True, False], [False, True]],
          valid=False)
      )
    ])
def test_roi_converter(origin, mask, expected):
    """
    Test method that converts ROIs to LIMS-like ROIs
    """
    actual = convert_to_lims_roi(origin, mask)
    assert actual == expected


def test_graph_to_img(example_graph):
    """
    smoke test graph_to_img
    """
    img = graph_to_img(example_graph,
                       attribute='dummy_attribute')

    # check that image has expected type and shape
    assert type(img) == np.ndarray
    assert img.shape == (40, 40)

    # check that ROI pixels are
    # brighter than non ROI pixels
    roi_mask = np.zeros((40, 40), dtype=bool)
    roi_mask[12:16, 4:7] = True
    roi_mask[25:32, 11:18] = True
    roi_mask[25:27, 15:18] = False

    roi_flux = img[roi_mask].flatten()
    complement = np.logical_not(roi_mask)
    not_roi_flux = img[complement].flatten()

    roi_mu = np.mean(roi_flux)
    roi_std = np.std(roi_flux, ddof=1)
    not_mu = np.mean(not_roi_flux)
    not_std = np.std(not_roi_flux, ddof=1)

    assert roi_mu > not_mu+roi_std+not_std


def test_find_peaks(example_img):
    """
    Test that find_peaks works with no mask
    """
    peaks = find_peaks(example_img, slop=2)
    assert len(peaks) == 2

    assert {'center': (2, 3),
            'rows': (0, 4),
            'cols': (1, 5)} in peaks

    assert {'center': (11, 12),
            'rows': (9, 13),
            'cols': (10, 14)} in peaks

    # test that, when the second peak is
    # masked, the third is found
    mask = np.zeros((20, 20), dtype=bool)
    mask[11, 12] = True
    peaks = find_peaks(example_img, mask=mask, slop=2)
    assert len(peaks) == 2

    assert {'center': (2, 3),
            'rows': (0, 4),
            'cols': (1, 5)} in peaks

    assert {'center': (10, 11),
            'rows': (8, 12),
            'cols': (9, 13)} in peaks


def test_caclulate_pearson_feature_vectors():
    """
    run smoke test on calculate_pearson_feature_vectors
    """
    rng = np.random.RandomState(491852)
    data = rng.random_sample((100, 20, 20))
    seed_pt = (15, 3)
    features = calculate_pearson_feature_vectors(
                                data,
                                seed_pt,
                                0.2)

    assert features.shape == (400, 400)

    # check that, if there is a mask, the appropriate
    # pixels are ignored
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:7, 11:] = True
    features = calculate_pearson_feature_vectors(
                                data,
                                seed_pt,
                                0.2,
                                pixel_ignore=mask)
    assert features.shape == (373, 373)


def test_segmenter(tmpdir, example_graph, example_video):
    """
    Smoke test for segmenter
    """

    segmenter = FeatureVectorSegmenter(graph_path=example_graph,
                                       video_path=example_video,
                                       attribute='dummy_attribute',
                                       filter_fraction=0.2,
                                        n_processors=1)

    dir_path = pathlib.Path(tmpdir)
    roi_path = dir_path / 'roi.json'
    seed_path = dir_path / 'seed.json'
    plot_path = dir_path / 'plot.png'
    assert not roi_path.exists()
    assert not seed_path.exists()
    assert not plot_path.exists()

    segmenter.run(roi_path=roi_path,
                  seed_path=seed_path,
                  plot_path=plot_path)

    assert roi_path.is_file()
    assert seed_path.is_file()
    assert plot_path.is_file()

    # check that some ROIs got written
    with open(roi_path, 'rb') as in_file:
        roi_data = json.load(in_file)
    assert len(roi_data) > 0

    # test that it can handle not receiving a
    # seed_path or plot_path
    roi_path.unlink()
    seed_path.unlink()
    plot_path.unlink()

    assert not roi_path.exists()
    assert not seed_path.exists()
    assert not plot_path.exists()

    segmenter.run(roi_path=roi_path,
                  seed_path=None,
                  plot_path=None)

    assert roi_path.is_file()
    assert not seed_path.exists()
    assert not plot_path.exists()


def test_segmenter_blank(tmpdir, blank_graph, blank_video):
    """
    Smoke test for segmenter on blank inputs
    """

    segmenter = FeatureVectorSegmenter(graph_path=blank_graph,
                                       video_path=blank_video,
                                       attribute='dummy_attribute',
                                       filter_fraction=0.2,
                                       n_processors=1)
    dir_path = pathlib.Path(tmpdir)
    roi_path = dir_path / 'roi.json'
    segmenter.run(roi_path=roi_path)
