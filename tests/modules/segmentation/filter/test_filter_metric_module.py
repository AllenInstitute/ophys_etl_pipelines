import pytest
import networkx
import pathlib
import numpy as np
import json
import h5py
from itertools import product

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    ophys_roi_to_extract_roi,
    deserialize_extract_roi_list)

from ophys_etl.modules.segmentation.filter.filter_utils import (
    mean_metric_from_roi,
    median_metric_from_roi)

from ophys_etl.modules.segmentation.modules.filter_metric_stat import (
    StatFilterRunner)

from ophys_etl.modules.segmentation.processing_log import (
    SegmentationProcessingLog)


@pytest.fixture(scope='session')
def graph_fixture(tmpdir_factory):
    tmpdir_path = pathlib.Path(tmpdir_factory.mktemp('stat_graph'))
    graph_path = tmpdir_path / 'stat_graph.pkl'

    graph = networkx.Graph()
    for r in range(32):
        for c in range(32):
            pt = (r, c)
            graph.add_node(pt)

    rng = np.random.default_rng(77123)
    for r in range(32):
        r0 = max(0, r-1)
        r1 = min(31, r+1)
        for c in range(32):
            c0 = max(0, c-1)
            c1 = min(31, c+1)
            pt = (r, c)
            for rr, cc in product(range(r0, r1), range(c0, c1)):
                if rr == r and cc == c:
                    continue
                other = (rr, cc)
                val = rng.random()
                graph.add_edge(pt, other, dummy_metric=val)

    networkx.write_gpickle(graph, graph_path)
    yield graph_path


@pytest.fixture(scope='session')
def roi_list_fixture(tmpdir_factory, graph_fixture):
    metric_img = graph_to_img(graph_fixture, attribute_name='dummy_metric')

    tmpdir_path = pathlib.Path(tmpdir_factory.mktemp('roi_list'))
    roi_list_path = tmpdir_path / 'roi_list.json'

    roi_list = []
    mean_lookup = dict()
    median_lookup = dict()

    rng = np.random.default_rng(623321)
    for ii in range(20):
        x0 = rng.integers(0, 20)
        y0 = rng.integers(0, 20)
        width = min(rng.integers(3, 6), 32-x0)
        height = min(rng.integers(3, 6), 32-y0)
        mask = rng.integers(0, 2, (height, width)).astype(bool)
        roi = OphysROI(x0=int(x0), width=int(width),
                       y0=int(y0), height=int(height),
                       valid_roi=True,
                       roi_id=ii,
                       mask_matrix=mask)
        mn = mean_metric_from_roi(roi, metric_img)
        md = median_metric_from_roi(roi, metric_img)
        mean_lookup[ii] = mn
        median_lookup[ii] = md

        roi_list.append(ophys_roi_to_extract_roi(roi))

    with open(roi_list_path, 'w') as out_file:
        out_file.write(json.dumps(roi_list, indent=2))

    yield {'path': roi_list_path,
           'mean_lookup': mean_lookup,
           'median_lookup': median_lookup}


@pytest.fixture(scope='function')
def processing_log_path_fixture(tmpdir, roi_list_fixture, seeder_fixture):
    with open(roi_list_fixture['path'], 'rb') as in_file:
        extract_roi_list = deserialize_extract_roi_list(
                                in_file.read())

    log_path = pathlib.Path(tmpdir) / 'roi_log_for_filter_test.h5'
    log = SegmentationProcessingLog(log_path, read_only=False)
    log.log_detection(attribute='dummy_metric',
                      rois=extract_roi_list,
                      group_name='detect',
                      seeder=seeder_fixture)
    yield log_path


@pytest.mark.parametrize(
    'stat_name, use_min, use_max',
    [('mean', False, True),
     ('mean', True, False),
     ('mean', True, True),
     ('median', False, True),
     ('median', True, False),
     ('median', True, True)])
def test_stat_filter_runner(processing_log_path_fixture,
                            graph_fixture, roi_list_fixture,
                            stat_name, use_min, use_max):

    lookup = roi_list_fixture[f'{stat_name}_lookup']
    value_list = list(lookup.values())
    value_list.sort()
    if use_min:
        min_val = value_list[4]
    else:
        min_val = None
    if use_max:
        max_val = value_list[16]
    else:
        max_val = None

    data = {'log_path': str(processing_log_path_fixture),
            'graph_input': str(graph_fixture),
            'pipeline_stage': 'just a test',
            'stat_name': stat_name,
            'attribute_name': 'dummy_metric',
            'min_value': min_val,
            'max_value': max_val}

    runner = StatFilterRunner(input_data=data, args=[])
    runner.run()
    with h5py.File(processing_log_path_fixture, 'r') as in_file:
        assert 'filter' in in_file.keys()
        filtered_rois = deserialize_extract_roi_list(
                             in_file['filter/rois'][()])
        log_invalid = set(in_file['filter/filter_ids'][()])
        log_reason = np.unique(in_file['filter/filter_reason'][()])

    expected_valid = set()
    expected_invalid = set()
    for roi_id in lookup:
        is_valid = True
        if min_val is not None and lookup[roi_id] < min_val:
            is_valid = False
        if max_val is not None and lookup[roi_id] > max_val:
            is_valid = False
        if is_valid:
            expected_valid.add(roi_id)
        else:
            expected_invalid.add(roi_id)
    assert len(expected_valid) > 0
    assert len(expected_invalid) > 0

    actual_valid = set([roi['id'] for roi in filtered_rois
                        if roi['valid']])
    actual_invalid = set([roi['id'] for roi in filtered_rois
                          if not roi['valid']])

    assert actual_valid == expected_valid
    assert actual_invalid == expected_invalid
    assert log_invalid == expected_invalid
    assert len(log_reason) == 1
    assert f'{stat_name}'.encode('utf-8') in log_reason[0]
