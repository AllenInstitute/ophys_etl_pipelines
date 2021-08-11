import pytest
import numpy as np
import pathlib
import json
import networkx
from itertools import product

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.utils.roi_utils import (
    ophys_roi_to_extract_roi)

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.segmentation.processing_log import (
    SegmentationProcessingLog)

from ophys_etl.modules.segmentation.seed.seeder import ImageMetricSeeder


@pytest.fixture
def seeder_fixture():
    # minimal seeder to satisfy logging
    seeder = ImageMetricSeeder()
    seeder._seed_image = np.zeros((10, 10))
    return seeder


@pytest.fixture(scope='session')
def area_roi_dict():
    """
    Create a dict of ROIs with IDs [1, 6)
    whose areas = 2*roi_id.

    The dict is keyed on roi_id.
    """

    rng = np.random.default_rng(66523)
    roi_dict = dict()

    for roi_id in range(1, 7, 1):
        x0 = rng.integers(0, 100)
        y0 = rng.integers(0, 100)
        width = rng.integers(10, 11)
        height = rng.integers(10, 15)
        mask = np.zeros(width*height).astype(bool)
        dexes = np.arange(len(mask), dtype=int)
        chosen = rng.choice(dexes, size=2*roi_id, replace=False)
        mask[chosen] = True
        roi = OphysROI(x0=int(x0), width=int(width),
                       y0=int(y0), height=int(height),
                       mask_matrix=mask.reshape((height, width)),
                       roi_id=roi_id,
                       valid_roi=True)
        assert roi.area == 2*roi_id
        roi_dict[roi_id] = roi
    return roi_dict


@pytest.fixture(scope='session')
def img_shape_fixture():
    """
    Tuple representing the shape of the image
    reference/created by the fixture below
    """
    return (64, 64)


@pytest.fixture(scope='session')
def roi_list_fixture(img_shape_fixture):
    """
    list of OphysROIs
    """
    rng = np.random.default_rng(771223)
    roi_list = []
    for ii in range(10):
        x0 = rng.integers(0, img_shape_fixture[1]-5)
        width = min(img_shape_fixture[1]-x0,
                    rng.integers(5, 8))
        y0 = rng.integers(0, img_shape_fixture[0]-5)
        height = min(img_shape_fixture[0]-y0,
                     rng.integers(5, 8))
        mask = rng.integers(0, 2, (height, width)).astype(bool)
        roi = OphysROI(x0=int(x0), width=int(width),
                       y0=int(y0), height=int(height),
                       valid_roi=True, roi_id=ii,
                       mask_matrix=mask)
        roi_list.append(roi)
    return roi_list


@pytest.fixture(scope='function')
def processing_log_path_fixture(tmpdir,
                                roi_list_fixture,
                                seeder_fixture):
    """
    Path to a processing log containing roi_list_fixture
    """
    extract_roi_list = [ophys_roi_to_extract_roi(roi)
                        for roi in roi_list_fixture]

    log_path = pathlib.Path(tmpdir) / 'roi_log_for_filter_test.h5'
    log = SegmentationProcessingLog(log_path, read_only=False)
    log.log_detection(attribute='dummy',
                      rois=extract_roi_list,
                      group_name='detect',
                      seeder=seeder_fixture)
    yield log_path


@pytest.fixture(scope='session')
def roi_list_path_fixture(tmpdir_factory, roi_list_fixture):
    """
    Path to a file where roi_list_fixture is serialized
    """
    tmpdir_path = pathlib.Path(tmpdir_factory.mktemp('z_score_roi'))
    roi_path = tmpdir_path / 'input_rois.json'
    extract_roi_list = [ophys_roi_to_extract_roi(roi)
                        for roi in roi_list_fixture]
    with open(roi_path, 'w') as out_file:
        out_file.write(json.dumps(extract_roi_list, indent=2))
    yield roi_path


@pytest.fixture(scope='session')
def graph_fixture(tmpdir_factory, roi_list_fixture, img_shape_fixture):
    """
    Graph used for generating a test image; metric is called 'dummy'.
    Pixels in ROIs from roi_list_fixture are correlated with each other
    """

    rng = np.random.default_rng(542392)
    graph = networkx.Graph()
    for r, c in product(range(img_shape_fixture[0]),
                        range(img_shape_fixture[1])):
        graph.add_node((r, c))
    for r, c in product(range(img_shape_fixture[0]),
                        range(img_shape_fixture[1])):
        r0 = max(0, r-1)
        r1 = min(r+2, img_shape_fixture[0])
        c0 = max(0, c-1)
        c1 = min(c+2, img_shape_fixture[1])
        for dr, dc in product(range(r0, r1), range(c0, c1)):
            if dr == 0 and dc == 0:
                continue
            graph.add_edge((r, c), (dr, dc),
                           dummy=rng.normal(0.22, 0.01))

    for roi in roi_list_fixture:
        mu = rng.random()*4.0 + roi.roi_id
        for i0 in range(roi.global_pixel_array.shape[0]):
            pt0 = (roi.global_pixel_array[i0, 0],
                   roi.global_pixel_array[i0, 1])
            for i1 in range(i0+1, roi.global_pixel_array.shape[1]):
                pt1 = (roi.global_pixel_array[i1, 0],
                       roi.global_pixel_array[i0, 1])
                edge = (pt0, pt1)
                graph.add_edge(edge[0], edge[1], dummy=rng.normal(mu, 0.1))
                graph.add_edge(edge[1], edge[0], dummy=rng.normal(mu, 0.1))

    graph_path = pathlib.Path(tmpdir_factory.mktemp('z_score_graph'))
    graph_path = graph_path / 'graph.pkl'
    networkx.write_gpickle(graph, graph_path)
    yield graph_path


@pytest.fixture(scope='session')
def img_fixture(graph_fixture):
    """
    Metric image derived from graph_fixture
    """
    return graph_to_img(graph_fixture,
                        attribute_name='dummy')
