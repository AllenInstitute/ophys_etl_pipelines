import json
import pytest

from ophys_etl.types import DenseROI
from ophys_etl.transforms.binarize_extract_bridge import \
        BinarizeToExtractBridge


@pytest.fixture
def mock_movie_path(tmp_path):
    mpath = tmp_path / "mock_movie.h5"
    with open(mpath, "w") as f:
        f.write("content")
    yield mpath


@pytest.fixture
def mock_compatible_path(tmp_path):
    cpath = tmp_path / "myrois.json"
    rois = []
    for i in range(10):
        rois.append(DenseROI(
            id=i+1,
            x=23,
            y=34,
            width=128,
            height=128,
            valid_roi=True,
            mask_matrix=[[True, True], [False, True]],
            max_correction_up=12,
            max_correction_down=12,
            max_correction_left=12,
            max_correction_right=12,
            mask_image_plane=0,
            exclusion_labels=['small_size', 'motion_border']))

    with open(cpath, "w") as f:
        json.dump(rois, f)
    yield cpath


@pytest.fixture
def mock_motion_csv(tmp_path):
    mpath = tmp_path / "m.csv"
    with open(mpath, "w") as fp:
        fp.write("content")
    yield mpath


def test_binarize_extract(mock_compatible_path, mock_motion_csv,
                          mock_movie_path, tmp_path):
    outpath = tmp_path / "myoutput.json"
    args = {
            'input_file': str(mock_compatible_path),
            'storage_directory': str(tmp_path),
            'motion_corrected_video': str(mock_movie_path),
            'motion_correction_values': str(mock_motion_csv),
            'output_json': str(outpath)
            }
    bridge = BinarizeToExtractBridge(input_data=args, args=[])
    bridge.run()

    with open(outpath, 'r') as f:
        outj = json.load(f)

    assert outj['log_0'] == str(mock_motion_csv)
    assert outj['motion_corrected_stack'] == str(mock_movie_path)
    assert outj['storage_directory'] == str(tmp_path)
    with open(mock_compatible_path, 'r') as f:
        nroi = len(json.load(f))
    assert len(outj['rois']) == nroi
    assert 'motion_border' in outj
