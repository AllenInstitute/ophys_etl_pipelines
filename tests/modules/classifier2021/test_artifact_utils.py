import pytest
import h5py
import json
import numpy as np
from itertools import product

from ophys_etl.modules.segmentation.utils.roi_utils import (
    sanitize_extract_roi_list,
    extract_roi_to_ophys_roi)

from ophys_etl.modules.classifier2021.utils import (
    get_traces,
    clip_img_to_quantiles)


def test_sanitize_extract_roi_list(
        suite2p_roi_fixture):

    with open(suite2p_roi_fixture, 'rb') as in_file:
        raw_roi_list = json.load(in_file)
    assert len(raw_roi_list) > 0
    extract_roi_list = sanitize_extract_roi_list(raw_roi_list)
    assert len(extract_roi_list) == len(raw_roi_list)
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


def test_get_traces(
        classifier2021_video_fixture,
        suite2p_roi_fixture):

    video_path = classifier2021_video_fixture
    roi_path = suite2p_roi_fixture

    with open(roi_path, 'rb') as in_file:
        raw_roi_list = json.load(in_file)
    extract_roi_list = sanitize_extract_roi_list(raw_roi_list)
    ophys_roi_list = [extract_roi_to_ophys_roi(roi)
                      for roi in extract_roi_list]

    found_traces = get_traces(video_path,
                              ophys_roi_list)

    assert len(found_traces) == len(extract_roi_list)

    with h5py.File(video_path, 'r') as in_file:
        video_data = in_file['data'][()]

    assert len(extract_roi_list) > 0
    for roi in extract_roi_list:
        assert roi['id'] in found_traces
        expected_trace = np.zeros(video_data.shape[0], dtype=float)
        npix = 0
        r0 = roi['y']
        c0 = roi['x']
        for ir in range(roi['height']):
            for ic in range(roi['width']):
                if not roi['mask'][ir][ic]:
                    continue
                npix += 1
                row = r0+ir
                col = c0+ic
                expected_trace += video_data[:, row, col]
        expected_trace = expected_trace/npix
        np.testing.assert_array_equal(expected_trace,
                                      found_traces[roi['id']])


@pytest.mark.parametrize("min_quantile,max_quantile",
                         product((0.1, 0.2), (0.7, 0.8)))
def test_clip_img_to_quantiles(min_quantile, max_quantile):
    rng = np.random.default_rng()
    img = rng.integers(0, 2**16-1, (100, 100)).astype(np.uint16)

    (lower_limit,
     upper_limit) = np.quantile(img, (min_quantile, max_quantile))

    clipped_img = clip_img_to_quantiles(
                      img,
                      (min_quantile, max_quantile))

    assert img.shape == clipped_img.shape
    preserved = 0
    eps = 1.0e-6
    for ii in range(img.shape[0]):
        for jj in range(img.shape[1]):
            raw_pixel = img[ii, jj]
            clipped_pixel = clipped_img[ii, jj]
            if raw_pixel < lower_limit:
                assert np.abs(clipped_pixel-lower_limit) < eps
            elif raw_pixel > upper_limit:
                assert np.abs(clipped_pixel-upper_limit) < eps
            else:
                assert np.abs(clipped_pixel-raw_pixel) < eps
                preserved += 1
    assert preserved > 0
