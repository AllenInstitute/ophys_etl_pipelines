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
        actual_invalid_roi = in_file['filter_log/roi_id'][()]
        actual_reason = in_file['filter_log/reason'][()]

    invalid_roi = set(roi_dict.keys()) - valid_roi
    for roi_id in invalid_roi:
        assert roi_id in actual_invalid_roi
    assert len(actual_invalid_roi) == len(actual_reason)

    expected = np.array(['area -- unit test'.encode('utf-8')])
    np.testing.assert_array_equal(
                np.unique(actual_reason),
                expected)
