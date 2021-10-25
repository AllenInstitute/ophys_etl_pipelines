import pytest
import h5py
import pathlib
import tempfile
import numpy as np
import json


@pytest.fixture(scope='session')
def classifier2021_tmpdir_fixture(tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('classifier2021'))
    yield pathlib.Path(tmpdir)


@pytest.fixture(scope='session')
def classifier2021_video_fixture(classifier2021_tmpdir_fixture):
    tmpdir = classifier2021_tmpdir_fixture
    video_path = tempfile.mkstemp(dir=tmpdir,
                                  prefix='classifier_video_',
                                  suffix='.h5')[1]
    video_path = pathlib.Path(video_path)
    rng = np.random.default_rng(71232213)
    data = rng.integers(0, (2**16)-1, (100, 40, 40)).astype(np.uint16)
    with h5py.File(video_path, 'w') as out_file:
        out_file.create_dataset('data', data=data)
    yield video_path


@pytest.fixture(scope='session')
def suite2p_roi_fixture(classifier2021_tmpdir_fixture):
    """
    Contains ROIs serialized according to the schema produced by our
    Suite2P segmentation pipeline ('mask_matrix' instead of 'mask',
    'valid_roi' instead of 'valid')
    """
    tmpdir = classifier2021_tmpdir_fixture
    roi_path = tempfile.mkstemp(dir=tmpdir,
                                prefix='classifier_rois_',
                                suffix='.json')[1]
    roi_path = pathlib.Path(roi_path)

    rng = np.random.default_rng(6242342)
    roi_list = []
    for roi_id in range(10):
        xx = int(rng.integers(0, 35))
        yy = int(rng.integers(0, 35))
        width = int(min(40-xx, rng.integers(5, 10)))
        height = int(min(40-yy, rng.integers(5, 10)))

        mask_sum = 0
        while mask_sum == 0:
            mask = [[bool(rng.integers(0,2)==0) for jj in range(width)]
                    for ii in range(height)]
            mask_sum = np.array(mask).sum()

        roi = dict()
        roi['roi_id'] = roi_id
        roi['valid_roi'] = True
        roi['x'] = xx
        roi['y'] = yy
        roi['width'] = width
        roi['height'] = height
        roi['mask_matrix'] = mask
        roi_list.append(roi)

    with open(roi_path, 'w') as out_file:
        out_file.write(json.dumps(roi_list, indent=2))
    yield roi_path
