import pytest
import pathlib
import numpy as np
import json

from ophys_etl.types import ExtractROI

from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysROI)

from ophys_etl.modules.segmentation.merge.roi_utils import (
    ophys_roi_to_extract_roi)

from ophys_etl.modules.segmentation.qc_utils.roi_comparison_utils import (
    roi_list_from_file)


@pytest.fixture(scope='session')
def list_of_roi():
    """
    A list of ExtractROIs
    """
    output = []
    rng = np.random.default_rng(11231)
    for ii in range(10):
        x0 = int(rng.integers(0, 1000))
        y0 = int(rng.integers(0, 1000))
        width = int(rng.integers(4, 10))
        height = int(rng.integers(4, 10))
        mask = rng.integers(0, 2, size=(height,width)).astype(bool)

        # because np.ints are not JSON serializable
        real_mask = []
        for row in mask:
            this_row = []
            for el in row:
                if el:
                    this_row.append(True)
                else:
                    this_row.append(False)
            real_mask.append(this_row)

        roi = ExtractROI(x=x0, width=width,
                         y=y0, height=height,
                         valid_roi=True,
                         mask=real_mask,
                         id=ii)
        output.append(roi)
    return output


@pytest.fixture(scope='session')
def roi_file(tmpdir_factory, list_of_roi):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('roi_reading'))
    file_path = tmpdir/'list_of_rois.json'
    with open(file_path, 'w') as out_file:
        json.dump(list_of_roi, out_file)
    yield file_path


def test_roi_list_from_file(roi_file, list_of_roi):
    raw_actual = roi_list_from_file(roi_file)
    actual = [ophys_roi_to_extract_roi(roi)
              for roi in raw_actual]
    assert actual == list_of_roi
