import pytest
import h5py
import pathlib
import tempfile
import numpy as np
import json
import PIL.Image
import networkx as nx
import hashlib
from itertools import combinations, product

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.utils.array_utils import normalize_array


def fixture_hasher(file_path):
    hasher = hashlib.md5()
    with open(file_path, 'rb') as in_file:
        chunk = in_file.read(1234567)
        while len(chunk) > 0:
            hasher.update(chunk)
            chunk = in_file.read(1234567)
    return hasher.hexdigest()


@pytest.fixture(scope='session')
def classifier2021_tmpdir_fixture(tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('classifier2021'))
    yield pathlib.Path(tmpdir)


@pytest.fixture(scope='session')
def classifier2021_video_fixture(classifier2021_tmpdir_fixture):
    tmpdir = classifier2021_tmpdir_fixture
    video_path = tempfile.mkstemp(dir=tmpdir,
                                  prefix='classifier_video_',
                                  suffix='.h5')[1]
    video_path = pathlib.Path(video_path)
    rng = np.random.default_rng(71232213)
    data = rng.integers(0, (2**16)-1, (100, 40, 40)).astype(np.uint16)
    with h5py.File(video_path, 'w') as out_file:
        out_file.create_dataset('data', data=data)
    yield video_path
    if video_path.is_file():
        video_path.unlink()


@pytest.fixture(scope='session')
def classifier2021_video_hash_fixture(
        classifier2021_video_fixture):
    return fixture_hasher(classifier2021_video_fixture)


@pytest.fixture(scope='session')
def suite2p_roi_fixture(classifier2021_tmpdir_fixture):
    """
    Contains ROIs serialized according to the schema produced by our
    Suite2P segmentation pipeline ('mask_matrix' instead of 'mask',
    'valid_roi' instead of 'valid')
    """
    tmpdir = classifier2021_tmpdir_fixture
    roi_path = tempfile.mkstemp(dir=tmpdir,
                                prefix='classifier_rois_',
                                suffix='.json')[1]
    roi_path = pathlib.Path(roi_path)

    rng = np.random.default_rng(6242342)
    roi_list = []
    for roi_id in range(10):
        xx = int(rng.integers(0, 35))
        yy = int(rng.integers(0, 35))
        width = int(min(40-xx, rng.integers(5, 10)))
        height = int(min(40-yy, rng.integers(5, 10)))

        mask_sum = 0
        while mask_sum == 0:
            mask = [[bool(rng.integers(0, 2) == 0) for jj in range(width)]
                    for ii in range(height)]
            mask_sum = np.array(mask).sum()

        roi = dict()
        roi['roi_id'] = roi_id
        roi['valid_roi'] = True
        roi['x'] = xx
        roi['y'] = yy
        roi['width'] = width
        roi['height'] = height
        roi['mask_matrix'] = mask
        roi_list.append(roi)

    with open(roi_path, 'w') as out_file:
        out_file.write(json.dumps(roi_list, indent=2))
    yield roi_path
    if roi_path.is_file():
        roi_path.unlink()


@pytest.fixture(scope='session')
def suite2p_roi_hash_fixture(
        suite2p_roi_fixture):
    return fixture_hasher(suite2p_roi_fixture)


@pytest.fixture(scope='session')
def classifier2021_corr_graph_fixture(
            classifier2021_tmpdir_fixture):
    tmpdir = classifier2021_tmpdir_fixture
    graph_path = tempfile.mkstemp(dir=tmpdir,
                                  prefix='corr_graph_',
                                  suffix='.pkl')[1]
    graph_path = pathlib.Path(graph_path)

    graph = nx.Graph()
    rng = np.random.default_rng(4422)
    coords = np.arange(0, 40)
    for xx, yy in combinations(coords, 2):
        minx = max(0, xx-1)
        miny = max(0, yy-1)
        maxx = min(xx+2, 40)
        maxy = min(yy+2, 40)
        xx_other = np.arange(minx, maxx)
        yy_other = np.arange(miny, maxy)
        for x1, y1 in product(xx_other, yy_other):
            graph.add_edge((xx, yy), (x1, y1),
                           filtered_hnc_Gaussian=rng.random())
    nx.write_gpickle(graph, graph_path)
    yield graph_path
    if graph_path.is_file():
        graph_path.unlink()


@pytest.fixture(scope='session')
def classifier2021_corr_graph_hash_fixture(
        classifier2021_corr_graph_fixture):
    return fixture_hasher(classifier2021_corr_graph_fixture)


@pytest.fixture(scope='session')
def classifier2021_corr_png_fixture(
        classifier2021_tmpdir_fixture,
        classifier2021_corr_graph_fixture):

    tmpdir = classifier2021_tmpdir_fixture
    png_path = tempfile.mkstemp(dir=tmpdir,
                                prefix='corr_png_',
                                suffix='.png')[1]

    img = graph_to_img(
                classifier2021_corr_graph_fixture,
                attribute_name='filtered_hnc_Gaussian')

    img = normalize_array(img)
    img = PIL.Image.fromarray(img)
    img.save(png_path)
    png_path = pathlib.Path(png_path)
    yield png_path
    if png_path.is_file():
        png_path.unlink()


@pytest.fixture(scope='session')
def classifier2021_corr_png_hash_fixture(
        classifier2021_corr_png_fixture):
    return fixture_hasher(classifier2021_corr_png_fixture)
