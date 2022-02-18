import pytest
import pathlib
import tempfile
import copy
import h5py
import json
import hashlib
import PIL.Image
import numpy as np
from itertools import product

from ophys_etl.utils.rois import (
    sanitize_extract_roi_list,
    extract_roi_to_ophys_roi)

from ophys_etl.modules.roi_cell_classifier.utils import (
    get_traces,
    clip_img_to_quantiles,
    create_metadata)


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


@pytest.fixture(scope='session')
def input_file_fixture(tmp_path_factory):
    """
    A dict mapping file type ('video', 'rois', etc.)
    to path and md5 hash
    """
    result = dict()
    rng = np.random.default_rng(72412)
    tmpdir = tmp_path_factory.mktemp('for_metadata_test')
    h5_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir, suffix='.h5')[1])
    with h5py.File(h5_path, 'w') as out_file:
        out_file.create_dataset('data', data=rng.random(100))
    h5_hash = hashlib.md5()
    with open(h5_path, 'rb') as in_file:
        h5_hash.update(in_file.read())
    h5_hash = h5_hash.hexdigest()
    result['video'] = {'path': str(h5_path.resolve().absolute()),
                       'hash': h5_hash}

    roi_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir, suffix='.json')[1])
    with open(roi_path, 'w') as out_file:
        out_file.write(json.dumps([{'hi': 'there'}, 5, 6.2], indent=2))
    roi_hash = hashlib.md5()
    with open(roi_path, 'rb') as in_file:
        roi_hash.update(in_file.read())
    roi_hash = roi_hash.hexdigest()
    result['rois'] = {'path': str(roi_path.resolve().absolute()),
                      'hash': roi_hash}

    correlation_path = pathlib.Path(tempfile.mkstemp(
                                         dir=tmpdir,
                                         suffix='.png')[1])
    img = PIL.Image.fromarray(rng.integers(0, 255, (32, 32)).astype(np.uint8))
    img.save(correlation_path)
    correlation_hash = hashlib.md5()
    with open(correlation_path, 'rb') as in_file:
        correlation_hash.update(in_file.read())
    correlation_hash = correlation_hash.hexdigest()
    correlation_path = str(correlation_path.resolve().absolute())
    result['correlation'] = {'path': correlation_path,
                             'hash': correlation_hash}

    motion_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir, suffix='.csv')[1])
    with open(motion_path, 'w') as out_file:
        out_file.write('qwertyuiopasdfghjkl')
    motion_hash = hashlib.md5()
    with open(motion_path, 'rb') as in_file:
        motion_hash.update(in_file.read())
    motion_hash = motion_hash.hexdigest()
    result['motion_csv'] = {'path': str(motion_path.resolve().absolute()),
                            'hash': motion_hash}

    yield result


@pytest.mark.parametrize('use_motion', (True, False))
def test_create_metadata(input_file_fixture, use_motion):
    """
    Test that create_metadata produces the expected output
    """

    input_args = {'a': 1, 'b': 2}
    file_metadata = copy.deepcopy(input_file_fixture)
    if use_motion:
        motion_path = pathlib.Path(file_metadata['motion_csv']['path'])
    else:
        file_metadata.pop('motion_csv')
        motion_path = None

    result = create_metadata(
                  input_args=input_args,
                  video_path=pathlib.Path(file_metadata['video']['path']),
                  roi_path=pathlib.Path(file_metadata['rois']['path']),
                  correlation_path=pathlib.Path(
                                       file_metadata['correlation']['path']),
                  motion_csv_path=motion_path)

    assert result['generator_args'] == input_args
    for k in file_metadata:
        assert result[k] == file_metadata[k]
    for k in result:
        if k == 'generator_args':
            continue
        assert k in file_metadata
