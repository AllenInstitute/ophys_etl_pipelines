import pytest
import h5py

from ophys_etl.modules.segmentation.utils.roi_utils import (
    background_mask_from_roi_list,
    deserialize_extract_roi_list)

from ophys_etl.modules.segmentation.filter.filter_utils import (
    z_vs_background_from_roi)

from ophys_etl.modules.segmentation.filter.roi_filter import (
    ZvsBackgroundFilter)

from ophys_etl.modules.segmentation.modules.filter_z_score import (
    ZvsBackgroundFilterRunner)


@pytest.fixture(scope='session')
def ground_truth_fixture(roi_list_fixture,
                         img_shape_fixture,
                         img_fixture):
    """
    lookup table mapping roi_id to z-score above background
    """
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
                        if roi['valid']])
    actual_invalid = set([roi['id'] for roi in results
                          if not roi['valid']])

    assert actual_valid == valid_roi_id
    assert actual_invalid == invalid_roi_id
