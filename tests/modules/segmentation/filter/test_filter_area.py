import pytest
import h5py
import numpy as np
import pathlib

from ophys_etl.modules.segmentation.modules.filter_area import (
    AreaFilterRunner)

from ophys_etl.modules.segmentation.qc_utils.roi_comparison_utils import (
    roi_list_from_file)

from ophys_etl.modules.decrosstalk.ophys_plane import compare_rois


@pytest.mark.parametrize(
        'min_area, max_area, valid_roi',
        [(4, 10, set([2, 3, 4, 5])),
         (None, 8, set([1, 2, 3, 4])),
         (6, None, set([3, 4, 5, 6]))
         ])
def test_area_filter_runner(tmpdir,
                            roi_dict,
                            roi_input_path,
                            min_area,
                            max_area,
                            valid_roi):
    dirpath = pathlib.Path(tmpdir)
    roi_log = dirpath/'roi_validity_log.h5'
    roi_output = dirpath/'roi_output.json'

    input_args = {'roi_input': str(roi_input_path),
                  'roi_output': str(roi_output),
                  'roi_log_path': str(roi_log),
                  'min_area': min_area,
                  'max_area': max_area,
                  'pipeline_stage': 'unit test'}

    runner = AreaFilterRunner(input_data=input_args,
                              args=[])
    runner.run()

    actual_rois = roi_list_from_file(roi_output)
    actual_lookup = {roi.roi_id: roi for roi in actual_rois}
    for roi_id in roi_dict:
        if roi_id in valid_roi:
            assert actual_lookup[roi_id].valid_roi
        else:
            assert not actual_lookup[roi_id].valid_roi
        assert compare_rois(actual_lookup[roi_id], roi_dict[roi_id])

    with h5py.File(roi_log, 'r') as in_file:
        actual_invalid_roi = in_file['filter_log/invalid_roi_id'][()]
        actual_reason = in_file['filter_log/reason'][()]

    invalid_roi = set(roi_dict.keys()) - valid_roi
    for roi_id in invalid_roi:
        assert roi_id in actual_invalid_roi
    assert len(actual_invalid_roi) == len(actual_reason)

    expected = np.array(['area -- unit test'.encode('utf-8')])
    np.testing.assert_array_equal(
                np.unique(actual_reason),
                expected)


def test_successive_area_filters(tmpdir,
                                 roi_dict,
                                 roi_input_path):
    dirpath = pathlib.Path(tmpdir)
    roi_min_cut_path = dirpath/'roi_min_cut.json'
    roi_max_cut_path = dirpath/'roi_max_cut.json'
    roi_log = dirpath/'roi_log.h5'

    # do a cut on minimum area; verify outputs
    min_cut_args = {'roi_input': str(roi_input_path),
                    'roi_output': str(roi_min_cut_path),
                    'roi_log_path': str(roi_log),
                    'min_area': 6,
                    'max_area': None,
                    'pipeline_stage': 'min cut'}

    runner = AreaFilterRunner(input_data=min_cut_args,
                              args=[])
    runner.run()

    min_cut_list = roi_list_from_file(roi_min_cut_path)
    invalid_roi = set([1, 2])
    assert len(min_cut_list) == len(roi_dict)
    for roi in min_cut_list:
        expected = roi_dict[roi.roi_id]
        assert compare_rois(roi, expected)
        if roi.roi_id in invalid_roi:
            assert not roi.valid_roi
        else:
            assert roi.valid_roi

    roi_id_set = set([roi.roi_id for roi in min_cut_list])
    assert roi_id_set == set(roi_dict.keys())

    with h5py.File(roi_log, 'r') as in_file:
        actual_invalid_roi = in_file['filter_log/invalid_roi_id'][()]
        actual_reason = in_file['filter_log/reason'][()]
    assert len(actual_invalid_roi) == 2
    assert 1 in actual_invalid_roi
    assert 2 in actual_invalid_roi
    assert len(actual_reason) == 2
    np.testing.assert_array_equal(
            np.unique(actual_reason),
            np.array(['area -- min cut'.encode('utf-8')]))

    # do a cut on maximum area using the output from the last
    # filter and writing to the same log
    max_cut_args = {'roi_input': str(roi_min_cut_path),
                    'roi_output': str(roi_max_cut_path),
                    'roi_log_path': str(roi_log),
                    'min_area': None,
                    'max_area': 8,
                    'pipeline_stage': 'max cut'}

    runner = AreaFilterRunner(input_data=max_cut_args,
                              args=[])
    runner.run()
    invalid_roi = set([1, 2, 5, 6])
    max_cut_list = roi_list_from_file(roi_max_cut_path)
    assert len(max_cut_list) == len(roi_dict)
    for roi in max_cut_list:
        expected = roi_dict[roi.roi_id]
        assert compare_rois(roi, expected)
        if roi.roi_id in invalid_roi:
            assert not roi.valid_roi
        else:
            assert roi.valid_roi

    roi_id_set = set([roi.roi_id for roi in max_cut_list])
    assert roi_id_set == set(roi_dict.keys())

    with h5py.File(roi_log, 'r') as in_file:
        actual_invalid_roi = in_file['filter_log/invalid_roi_id'][()]
        actual_reason = in_file['filter_log/reason'][()]
    assert set(actual_invalid_roi) == invalid_roi
    assert len(actual_invalid_roi) == len(invalid_roi)
    assert len(actual_reason) == len(invalid_roi)
    for roi_id, reason in zip(invalid_roi, actual_reason):
        if roi_id in (1, 2):
            assert reason == 'area -- min cut'.encode('utf-8')
        else:
            assert reason == 'area -- max cut'.encode('utf-8')
