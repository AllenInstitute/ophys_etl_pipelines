import pytest
import numpy as np
import h5py

from ophys_etl.modules.segmentation.utils.roi_utils import (
    deserialize_extract_roi_list,
    mean_metric_from_roi,
    median_metric_from_roi)

from ophys_etl.modules.segmentation.modules.filter_metric_stat import (
    StatFilterRunner)


@pytest.fixture(scope='session')
def mean_lookup_fixture(img_fixture, roi_list_fixture):
    lookup = dict()
    for roi in roi_list_fixture:
        lookup[roi.roi_id] = mean_metric_from_roi(roi, img_fixture)
    return lookup


@pytest.fixture(scope='session')
def median_lookup_fixture(img_fixture, roi_list_fixture):
    lookup = dict()
    for roi in roi_list_fixture:
        lookup[roi.roi_id] = median_metric_from_roi(roi, img_fixture)
    return lookup


@pytest.mark.parametrize(
    'stat_name, use_min, use_max',
    [('mean', False, True),
     ('mean', True, False),
     ('mean', True, True),
     ('median', False, True),
     ('median', True, False),
     ('median', True, True)])
def test_stat_filter_runner(processing_log_path_fixture,
                            graph_fixture,
                            median_lookup_fixture,
                            mean_lookup_fixture,
                            stat_name, use_min, use_max):

    if stat_name == 'mean':
        lookup = mean_lookup_fixture
    elif stat_name == 'median':
        lookup = median_lookup_fixture
    else:
        raise RuntimeError(f'test cannot parse stat {stat_name}')

    value_list = list(lookup.values())
    value_list.sort()
    if use_min:
        min_val = value_list[4]
    else:
        min_val = None
    if use_max:
        max_val = value_list[8]
    else:
        max_val = None

    data = {'log_path': str(processing_log_path_fixture),
            'graph_input': str(graph_fixture),
            'pipeline_stage': 'just a test',
            'stat_name': stat_name,
            'attribute_name': 'dummy',
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
