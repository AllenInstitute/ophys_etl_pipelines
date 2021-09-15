import pytest
import h5py
import pathlib
import datetime
import contextlib
import numpy as np
import matplotlib

from ophys_etl.types import ExtractROI
from ophys_etl.modules.segmentation import processing_log
from ophys_etl.modules.segmentation.seed.seeder import ImageMetricSeeder
from ophys_etl.modules.segmentation.utils.roi_utils import (
    serialize_extract_roi_list)


@pytest.fixture
def extract_roi_list():
    nroi = 20
    xs = np.linspace(start=10, stop=490, num=nroi, dtype=int)
    ys = np.linspace(start=20, stop=470, num=nroi, dtype=int)
    widths = ([10] * (nroi - 10) + [12] * 10)
    heights = ([8] * (nroi - 7) + [11] * 7)
    valids = ([True] * (nroi - 4) + [False] * 4)

    rng = np.random.default_rng(3342)
    roi_list = []
    for i in range(nroi):
        mask = [row.tolist()
                for row in rng.integers(low=0,
                                        high=2,
                                        size=(heights[i], widths[i])
                                        ).astype(bool)]
        roi_list.append(
                ExtractROI(
                    id=int((i + 1)),
                    x=int(xs[i]),
                    y=int(ys[i]),
                    width=int(widths[i]),
                    height=int(heights[i]),
                    mask=mask,
                    valid=valids[i]))
    return roi_list


@pytest.fixture
def seeder_fixture():
    # minimal seeder to satisfy logging
    seeder = ImageMetricSeeder()
    seeder._seed_image = np.zeros((512, 512)).astype("uint8")
    return seeder


def test_timestamp(tmpdir):
    h5path = tmpdir / "test.h5"
    with h5py.File(h5path, "w") as f:
        group = f.create_group("timestamp_test")
        processing_log.timestamp_group(group)
        assert "group_creation_time" in group
        dstr = group["group_creation_time"][()].decode("utf-8")

    assert isinstance(datetime.datetime.fromisoformat(dstr),
                      datetime.datetime)


@pytest.mark.parametrize(
        "read_only, context",
        [
            (True, pytest.warns(
                UserWarning, match=r".*can not be invoked.*")),
            (False, contextlib.nullcontext())
            ])
def test_read_only(tmpdir, extract_roi_list,
                   seeder_fixture, read_only, context):
    """tests that write methods emit a warning and don't write
    when read_only=False
    """
    h5path = tmpdir / "test.h5"
    plog = processing_log.SegmentationProcessingLog(path=h5path,
                                                    read_only=read_only)
    assert plog.read_only == read_only

    # log_detect should emit a warning and not create a step
    # if SegmentationProcessingLog.read_only=True
    with context:
        plog.log_detection(attribute="attribute",
                           rois=extract_roi_list,
                           group_name="detect",
                           seeder=seeder_fixture,
                           seeder_group_name="seed")
    if read_only:
        assert "processing_steps" not in plog.__dict__
    else:
        assert plog.get_last_group() == "detect"

    # log_merge should emit a warning and not create a step
    # if SegmentationProcessingLog.read_only=True
    with context:
        plog.log_merge(merger_ids=[],
                       rois=extract_roi_list,
                       roi_source_group="detect",
                       group_name="merge")
    if read_only:
        assert "processing_steps" not in plog.__dict__
    else:
        assert plog.get_last_group() == "merge"

    # log_filter should emit a warning and not create a step
    # if SegmentationProcessingLog.read_only=True
    with context:
        plog.log_filter(filter_ids=[],
                        filter_reason="why not?",
                        rois=extract_roi_list,
                        roi_source_group="merge",
                        group_name="filter")
    if read_only:
        assert "processing_steps" not in plog.__dict__
    else:
        assert plog.get_last_group() == "filter"


def test_SegmentationProcessingLog_init():
    path = "abc.xyz"
    plog = processing_log.SegmentationProcessingLog(path=path)
    assert plog.path == path
    assert plog.steps_dataset_name == "processing_steps"
    assert plog.read_only


def test_get_detection_group(tmpdir):
    path = tmpdir / "log.h5"
    plog = processing_log.SegmentationProcessingLog(path=path)
    assert plog.get_detection_group() is None

    with h5py.File(path, "w") as f:
        f.create_dataset("something_else_but_file_exists", data=[])
    assert plog.get_detection_group() is None

    with h5py.File(path, "w") as f:
        f.create_dataset("detection_group", data="testme".encode("utf-8"))
    assert plog.get_detection_group() == "testme"


def test_append_processing_step(tmpdir):
    path = tmpdir / "log.h5"
    plog = processing_log.SegmentationProcessingLog(path=path,
                                                    read_only=False)
    plog._append_processing_step("test")
    with h5py.File(plog.path, "r") as f:
        steps = [i.decode("utf-8") for i in f[plog.steps_dataset_name][()]]
    assert steps == ["test"]

    plog._append_processing_step("test")
    with h5py.File(plog.path, "r") as f:
        steps = [i.decode("utf-8") for i in f[plog.steps_dataset_name][()]]
    assert steps == ["test", "test_1"]


def test_log_two_detection_attempts(tmpdir, extract_roi_list, seeder_fixture):
    """do not allow more than one detection group per file
    """
    path = tmpdir / "log.h5"
    plog = processing_log.SegmentationProcessingLog(path=path,
                                                    read_only=False)
    plog.log_detection(attribute="attribute",
                       rois=extract_roi_list,
                       group_name="detect",
                       seeder=seeder_fixture,
                       seeder_group_name="seed")
    with pytest.raises(processing_log.SegmentationProcessingLogError,
                       match="there is already a detection_group"):
        plog.log_detection(attribute="attribute",
                           rois=extract_roi_list,
                           group_name="detect",
                           seeder=seeder_fixture,
                           seeder_group_name="seed")


@pytest.mark.parametrize('only_valid', [True, False])
def test_create_figures(tmpdir, extract_roi_list, seeder_fixture, only_valid):
    """smoke test
    """
    h5path = tmpdir / "test.h5"
    plog = processing_log.SegmentationProcessingLog(path=h5path,
                                                    read_only=False)
    plog.log_detection(attribute="attribute",
                       rois=extract_roi_list,
                       group_name="detect",
                       seeder=seeder_fixture,
                       seeder_group_name="seed")
    fig = plog.create_seeder_figure()
    assert isinstance(fig, matplotlib.figure.Figure)

    fig = plog.create_roi_metric_figure(only_valid=only_valid)
    assert isinstance(fig, matplotlib.figure.Figure)

    plog.log_merge(rois=extract_roi_list,
                   roi_source_group="detect",
                   merger_ids=[])
    fig = plog.create_roi_merge_figure()
    assert isinstance(fig, matplotlib.figure.Figure)


@pytest.mark.parametrize('valid_only', [True, False])
def test_get_rois(tmpdir, extract_roi_list, valid_only):
    log_path = pathlib.Path(tmpdir) / 'dummy_test_log.h5'
    with h5py.File(log_path, 'w') as out_file:
        out_file.create_dataset(
                'detect/rois',
                data=serialize_extract_roi_list(extract_roi_list))

    splog = processing_log.SegmentationProcessingLog(
                    log_path,
                    read_only=True)

    expected_list = []
    for roi in extract_roi_list:
        if not valid_only or roi['valid']:
            expected_list.append(roi)

    actual = splog.get_rois_from_group('detect', valid_only=valid_only)
    assert actual == expected_list

    expected_dict = {roi['id']: roi for roi in expected_list}
    actual = splog.get_roi_lookup_from_group('detect', valid_only=valid_only)
    assert actual == expected_dict
