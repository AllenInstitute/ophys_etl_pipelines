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
    ophys_roi_to_extract_roi)

from ophys_etl.modules.segmentation.filter.filter_utils import (
    mean_metric_from_roi,
    median_metric_from_roi)

from ophys_etl.modules.segmentation.modules.filter_metric_stat import (
    StatFilterRunner)


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


@pytest.mark.parametrize(
    'stat_name, use_min, use_max',
    [('mean', False, True),
     ('mean', True, False),
     ('mean', True, True),
     ('median', False, True),
     ('median', True, False),
     ('median', True, True)])
def test_stat_filter_runner(tmpdir,
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

    tmpdir_path = pathlib.Path(tmpdir)
    out_path = tmpdir_path/'output_rois.json'
    log_path = tmpdir_path/'log.h5'
    data = {'roi_input': str(roi_list_fixture['path']),
            'graph_input': str(graph_fixture),
            'roi_output': str(out_path),
            'roi_log_path': str(log_path),
            'pipeline_stage': 'just a test',
            'stat_name': stat_name,
            'attribute_name': 'dummy_metric',
            'min_value': min_val,
            'max_value': max_val}

    runner = StatFilterRunner(input_data=data, args=[])
    runner.run()
    assert out_path.is_file()
    assert log_path.is_file()

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
    with open(out_path, 'rb') as in_file:
        filtered_rois = json.load(in_file)
    actual_valid = set([roi['id'] for roi in filtered_rois
                        if roi['valid_roi']])
    actual_invalid = set([roi['id'] for roi in filtered_rois
                          if not roi['valid_roi']])

    assert actual_valid == expected_valid
    assert actual_invalid == expected_invalid

    with h5py.File(log_path, 'r') as in_file:
        qc_log = in_file['filter_log']
        log_invalid = set(qc_log['invalid_roi_id'][()])
        log_reason = np.unique(qc_log['reason'][()])
    assert log_invalid == expected_invalid
    assert len(log_reason) == 1
    assert f'{stat_name}'.encode('utf-8') in log_reason[0]
