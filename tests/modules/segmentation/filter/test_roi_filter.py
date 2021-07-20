import h5py
import pytest
import numpy as np
import pathlib

from ophys_etl.modules.segmentation.filter.roi_filter import (
    ROIBaseFilter,
    ROIAreaFilter,
    log_invalid_rois)

from ophys_etl.modules.segmentation.filter.schemas import (
    AreaFilterSchema)


def test_base_roi_filter():

    class DummyFilter(ROIBaseFilter):

        def is_roi_valid(self, roi):
            return {'valid_roi': [],
                    'invalid_roi': []}

    roi_filter = DummyFilter()

    with pytest.raises(NotImplementedError,
                       match="self._reason not defined"):
        roi_filter.reason


def test_invalid_area_roi_filter(roi_dict):
    with pytest.raises(RuntimeError, match='Both max_area and min_area'):
        ROIAreaFilter()


@pytest.mark.parametrize(
        'min_area, max_area, expected_valid',
        [(None, 8, set([1, 2, 3, 4])),
         (6, None, set([3, 4, 5, 6])),
         (4, 8, set([2, 3, 4]))
         ])
def test_area_roi_filter(roi_dict, min_area, max_area, expected_valid):

    area_filter = ROIAreaFilter(min_area=min_area,
                                max_area=max_area)

    assert area_filter.reason == 'area'

    results = area_filter.do_filtering(list(roi_dict.values()))

    valid_lookup = {roi.roi_id: roi for roi in results['valid_roi']}
    invalid_lookup = {roi.roi_id: roi for roi in results['invalid_roi']}
    assert (len(results['valid_roi'])
            + len(results['invalid_roi'])) == len(roi_dict)

    for roi_id in roi_dict:
        if roi_id in expected_valid:
            assert roi_id in valid_lookup
            actual_roi = valid_lookup[roi_id]
            assert actual_roi.valid_roi
        else:
            assert roi_id in invalid_lookup
            actual_roi = invalid_lookup[roi_id]
            assert not actual_roi.valid_roi

        expected_roi = roi_dict[roi_id]
        assert expected_roi.x0 == actual_roi.x0
        assert expected_roi.y0 == actual_roi.y0
        assert expected_roi.width == actual_roi.width
        assert expected_roi.height == actual_roi.height
        np.testing.assert_array_equal(expected_roi.mask_matrix,
                                      actual_roi.mask_matrix)


def test_area_filter_schema(tmpdir):
    area_schema = AreaFilterSchema()
    log_file_path = pathlib.Path(tmpdir)/'dummy_log.h5'
    input_json = str(pathlib.Path(tmpdir)/'input.json')
    output_json = str(pathlib.Path(tmpdir)/'output.json')
    with open(input_json, 'w') as out_file:
        out_file.write('hi there')

    valid_schema = {'roi_log_path': str(log_file_path.absolute()),
                    'pipeline_stage': 'something',
                    'roi_input': input_json,
                    'roi_output': output_json,
                    'max_area': 5,
                    'min_area': 3}

    area_schema.load(valid_schema)

    valid_schema = {'roi_log_path': str(log_file_path.absolute()),
                    'pipeline_stage': 'something',
                    'roi_input': input_json,
                    'roi_output': output_json,
                    'max_area': 5,
                    'min_area': None}

    area_schema.load(valid_schema)

    valid_schema = {'roi_log_path': str(log_file_path.absolute()),
                    'pipeline_stage': 'something',
                    'roi_input': input_json,
                    'roi_output': output_json,
                    'max_area': None,
                    'min_area': 2}

    area_schema.load(valid_schema)

    with pytest.raises(RuntimeError, match='are both None'):
        invalid_schema = {'roi_log_path': str(log_file_path.absolute()),
                          'pipeline_stage': 'something',
                          'roi_input': input_json,
                          'roi_output': output_json,
                          'max_area': None,
                          'min_area': None}

        area_schema.load(invalid_schema)


def test_log_invalid_rois(tmpdir, roi_dict):

    # test case of non-empty log file
    log_path = pathlib.Path(tmpdir)/'dummy_invalid_log.h5'
    with h5py.File(log_path, 'w') as out_file:
        out_file.create_dataset('unrelated_data', data=np.arange(20, 30, 1))
        g = out_file.create_group('filter_log')
        g.create_dataset('roi_id', data=np.arange(-5, -1, 1))
        g.create_dataset('reason',
                         data=np.array(['prefill'.encode('utf-8')]*4))

    roi_list = list(roi_dict.values())
    log_invalid_rois(roi_list, 'area post-merge', log_path)

    expected_roi_id = [-5, -4, -3, -2] + list([roi.roi_id for roi in roi_list])
    expected_roi_id = np.array(expected_roi_id)
    expected_reasons = ['prefill'.encode('utf-8')]*4
    expected_reasons += ['area post-merge'.encode('utf-8')]*6
    expected_reasons = np.array(expected_reasons)
    assert len(expected_reasons) == len(expected_roi_id)

    with h5py.File(log_path, 'r') as in_file:
        actual_roi_id = in_file['filter_log/roi_id'][()]
        actual_reasons = in_file['filter_log/reason'][()]
        np.testing.assert_array_equal(in_file['unrelated_data'][()],
                                      np.arange(20, 30, 1))
    np.testing.assert_array_equal(actual_roi_id, expected_roi_id)
    np.testing.assert_array_equal(actual_reasons, expected_reasons)

    # test case of empty log file
    log_path = pathlib.Path(tmpdir)/'dummy_invalid_log2.h5'
    roi_list = list(roi_dict.values())
    log_invalid_rois(roi_list, 'area pre-merge', log_path)

    with h5py.File(log_path, 'r') as in_file:
        actual_roi_id = in_file['filter_log/roi_id'][()]
        actual_reasons = in_file['filter_log/reason'][()]

    np.testing.assert_array_equal(actual_roi_id,
                                  np.array([roi.roi_id for roi in roi_list]))
    np.testing.assert_array_equal(
            actual_reasons,
            np.array(['area pre-merge'.encode('utf-8')]*len(roi_list)))
