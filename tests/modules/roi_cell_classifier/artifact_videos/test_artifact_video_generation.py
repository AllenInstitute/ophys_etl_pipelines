import pytest
import h5py
import pathlib
import hashlib
import numpy as np
import copy
from itertools import product
from ophys_etl.utils.thumbnail_video_generator import (
    VideoGenerator)
from ophys_etl.modules.roi_cell_classifier.video_utils import (
    get_thumbnail_video_from_artifact_file)


def path_to_hash(this_path):
    """
    Return the md5 checksum of a file specified by a path
    """
    hasher = hashlib.md5()
    with open(this_path, 'rb') as in_file:
        chunk = in_file.read(10000)
        while len(chunk) > 0:
            hasher.update(chunk)
            chunk = in_file.read(100000)
    return hasher.hexdigest()


@pytest.mark.parametrize(
    "padding, use_other_roi, roi_color, timesteps",
    product((0, 5), (True, False),
            (None, (122, 11, 6)),
            (None, np.arange(6, 36, dtype=int))))
def test_video_gen(
        video_file_fixture,
        extract_roi_list_fixture,
        tmp_path_factory,
        padding,
        use_other_roi,
        roi_color,
        timesteps):
    """
    Test that get_thumbnail_video_from_artifact_file returns
    the same results as VideoGenerator.get_thumbnail_video_from_roi
    """
    tmp_dir = pathlib.Path(tmp_path_factory.mktemp('roundtrip'))
    alt_tmp_dir = pathlib.Path(tmp_path_factory.mktemp('by_hand'))

    with h5py.File(video_file_fixture, 'r') as in_file:
        data_arr = in_file['video_data'][()]
    gen0 = VideoGenerator(video_data=data_arr,
                          tmp_dir=tmp_dir)

    for ii, roi in enumerate(extract_roi_list_fixture):
        if use_other_roi:
            other_roi = copy.deepcopy(extract_roi_list_fixture)
            other_roi.pop(ii)
        else:
            other_roi = None

        v0 = gen0.get_thumbnail_video_from_roi(
                                   roi=roi,
                                   padding=padding,
                                   other_roi=other_roi,
                                   roi_color=roi_color,
                                   timesteps=timesteps)

        v1 = get_thumbnail_video_from_artifact_file(
                    artifact_path=video_file_fixture,
                    roi=roi,
                    padding=padding,
                    other_roi=other_roi,
                    roi_color=roi_color,
                    timesteps=timesteps,
                    tmp_dir=alt_tmp_dir)

        h0 = path_to_hash(v0.video_path)
        h1 = path_to_hash(v1.video_path)
        assert h0 == h1
        del v0
        del v1
