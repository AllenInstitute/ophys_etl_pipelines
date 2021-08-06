import pytest
import h5py
import datetime
import contextlib
import numpy as np

from ophys_etl.types import ExtractROI
from ophys_etl.modules.segmentation import processing_log
from ophys_etl.modules.segmentation.seed.seeder import ImageMetricSeeder


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
    seeder._seed_image = np.zeros((10, 10))
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

    # log_detection should be read only when asked
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

    # log_merge should be read only when asked
    with context:
        plog.log_merge(merger_ids=[],
                       rois=extract_roi_list,
                       group_name="merge")
    if read_only:
        assert "processing_steps" not in plog.__dict__
    else:
        assert plog.get_last_group() == "merge"

    # log_filter should be read only when asked
    with context:
        plog.log_filter(filter_ids=[],
                        filter_reason="why not?",
                        rois=extract_roi_list,
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
