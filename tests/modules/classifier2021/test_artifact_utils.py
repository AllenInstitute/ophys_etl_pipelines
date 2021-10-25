import json

from ophys_etl.modules.segmentation.utils.roi_utils import (
    sanitize_extract_roi_list)


def test_sanitize_extract_roi_list(
        suite2p_roi_fixture):

    with open(suite2p_roi_fixture, 'rb') as in_file:
        raw_roi_list = json.load(in_file)
    extract_roi_list = sanitize_extract_roi_list(raw_roi_list)
    for raw_roi, extract_roi in zip(raw_roi_list,
                                    extract_roi_list):

        assert 'mask_matrix' not in extract_roi
        assert 'valid_roi' not in extract_roi
        assert 'roi_id' not in extract_roi
        for e_key, r_key in (('id', 'roi_id'),
                             ('mask', 'mask_matrix'),
                             ('valid', 'valid_roi'),
                             ('x', 'x'),
                             ('y', 'y'),
                             ('width', 'width'),
                             ('height', 'height')):

            assert extract_roi[e_key] == raw_roi[r_key]
