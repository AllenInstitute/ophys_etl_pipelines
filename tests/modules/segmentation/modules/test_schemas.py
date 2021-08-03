import pytest
import h5py
from pathlib import Path
from marshmallow import ValidationError

from ophys_etl.modules.segmentation.modules import schemas


@pytest.fixture
def dummy_video_file(tmpdir):
    h5path = Path(tmpdir / "dummy_video.h5")
    h5path.touch()
    yield str(h5path)


def test_roi_merger_schema_rois_hdf5(tmpdir, dummy_video_file):
    # has no group 'detect'
    h5path = Path(tmpdir / "test.h5")
    roipath = Path(tmpdir / "test.json")
    roipath.touch()
    with h5py.File(h5path, "w") as f:
        f.create_dataset("something_else", data=[1, 2, 3])
    args = {
            "video_input": dummy_video_file,
            "roi_input": None,
            "qc_output": str(h5path),
            }
    with pytest.raises(ValidationError,
                       match=r"must contain the group 'detect'"):
        schemas.RoiMergerSchema().load(data=args)
    # 'roi_input' satisfies check
    args['roi_input'] = str(roipath)
    schemas.RoiMergerSchema().load(data=args)

    # has group 'detect' but no dataset 'rois'
    args['roi_input'] = None
    with h5py.File(h5path, "w") as f:
        f.create_group("detect")
    with pytest.raises(ValidationError,
                       match=r"must contain the dataset 'rois'"):
        schemas.RoiMergerSchema().load(data=args)
    # 'roi_input' satisfies check
    args['roi_input'] = str(roipath)
    schemas.RoiMergerSchema().load(data=args)
