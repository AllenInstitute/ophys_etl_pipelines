import pytest
import h5py
import numpy as np
from itertools import product
import pathlib
import json
import networkx

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.utils.roi_utils import (
    ophys_roi_to_extract_roi)

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    background_mask_from_roi_list,
    deserialize_extract_roi_list)

from ophys_etl.modules.segmentation.filter.filter_utils import (
    z_vs_background_from_roi)

from ophys_etl.modules.segmentation.filter.roi_filter import (
    ZvsBackgroundFilter)

from ophys_etl.modules.segmentation.modules.filter_z_score import (
    ZvsBackgroundFilterRunner)

from ophys_etl.modules.segmentation.processing_log import (
    SegmentationProcessingLog)

from ophys_etl.modules.segmentation.seed.seeder import ImageMetricSeeder


@pytest.fixture(scope='session')
def img_shape_fixture():
    return (64, 64)


@pytest.fixture(scope='session')
def roi_list_fixture(img_shape_fixture):
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


@pytest.fixture
def seeder_fixture():
    # minimal seeder to satisfy logging
    seeder = ImageMetricSeeder()
    seeder._seed_image = np.zeros((10, 10))
    return seeder


@pytest.fixture(scope='function')
def processing_log_path_fixture(tmpdir, roi_list_fixture, seeder_fixture):
    extract_roi_list = [ophys_roi_to_extract_roi(roi)
                        for roi in roi_list_fixture]

    log_path = pathlib.Path(tmpdir) / 'roi_log_for_filter_test.h5'
    log = SegmentationProcessingLog(log_path, read_only=False)
    log.log_detection(attribute='dummy_metric',
                      rois=extract_roi_list,
                      group_name='detect',
                      seeder=seeder_fixture)
    yield log_path


@pytest.fixture(scope='session')
def roi_list_path_fixture(tmpdir_factory, roi_list_fixture):
    tmpdir_path = pathlib.Path(tmpdir_factory.mktemp('z_score_roi'))
    roi_path = tmpdir_path / 'input_rois.json'
    extract_roi_list = [ophys_roi_to_extract_roi(roi)
                        for roi in roi_list_fixture]
    with open(roi_path, 'w') as out_file:
        out_file.write(json.dumps(extract_roi_list, indent=2))
    yield roi_path


@pytest.fixture(scope='session')
def graph_fixture(tmpdir_factory, roi_list_fixture, img_shape_fixture):

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
    return graph_to_img(graph_fixture,
                        attribute_name='dummy')


@pytest.fixture(scope='session')
def ground_truth_fixture(roi_list_fixture,
                         img_shape_fixture,
                         img_fixture):
    bckgd_mask = background_mask_from_roi_list(
                       roi_list_fixture,
                       img_shape_fixture)

    z_score_lookup = dict()
    for roi in roi_list_fixture:
        z_score = z_vs_background_from_roi(
                      roi,
                      img_fixture,
                      bckgd_mask,
                      0.25,
                      n_desired_background=max(15, 2*roi.area))
        z_score_lookup[roi.roi_id] = z_score
    return z_score_lookup


@pytest.mark.parametrize('cutoff', [0.0, 9.0, 24.0, 25.0])
def test_z_vs_background_filter(
        roi_list_fixture,
        img_fixture,
        ground_truth_fixture,
        cutoff):

    valid_roi_id = set()
    invalid_roi_id = set()
    for roi in roi_list_fixture:
        if ground_truth_fixture[roi.roi_id] < cutoff:
            invalid_roi_id.add(roi.roi_id)
        else:
            valid_roi_id.add(roi.roi_id)

    this_filter = ZvsBackgroundFilter(
                     img_fixture,
                     cutoff,
                     2,
                     15,
                     0.25)
    results = this_filter.do_filtering(roi_list_fixture)

    actual_valid = set([r.roi_id for r in results['valid_roi']])
    actual_invalid = set([r.roi_id for r in results['invalid_roi']])
    assert actual_valid == valid_roi_id
    assert actual_invalid == invalid_roi_id


@pytest.mark.parametrize('cutoff', [0.0, 9.0, 24.0, 25.0])
def test_z_vs_background_module(
        processing_log_path_fixture,
        roi_list_path_fixture,
        roi_list_fixture,
        graph_fixture,
        ground_truth_fixture,
        cutoff):

    valid_roi_id = set()
    invalid_roi_id = set()
    for roi in roi_list_fixture:
        if ground_truth_fixture[roi.roi_id] < cutoff:
            invalid_roi_id.add(roi.roi_id)
        else:
            valid_roi_id.add(roi.roi_id)

    data = {'log_path': str(processing_log_path_fixture),
            'pipeline_stage': 'test filter',
            'graph_input': str(graph_fixture),
            'attribute_name': 'dummy',
            'min_z': cutoff,
            'n_background_factor': 2,
            'n_background_minimum': 15}

    runner = ZvsBackgroundFilterRunner(
                   input_data=data,
                   args=[])
    runner.run()

    with h5py.File(processing_log_path_fixture, 'r') as in_file:
        results = deserialize_extract_roi_list(
                             in_file['filter/rois'][()])

    actual_valid = set([roi['id'] for roi in results
                        if roi['valid_roi']])
    actual_invalid = set([roi['id'] for roi in results
                          if not roi['valid_roi']])

    assert actual_valid == valid_roi_id
    assert actual_invalid == invalid_roi_id
