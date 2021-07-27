import pytest
import numpy as np
import pathlib
import json
from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI


@pytest.fixture
def example_roi_list():
    rng = np.random.RandomState(6412439)
    roi_list = []
    for ii in range(30):
        x0 = rng.randint(0, 25)
        y0 = rng.randint(0, 25)
        height = rng.randint(3, 7)
        width = rng.randint(3, 7)
        mask = rng.randint(0, 2, (height, width)).astype(bool)
        roi = OphysROI(x0=x0, y0=y0,
                       height=height, width=width,
                       mask_matrix=mask,
                       roi_id=ii,
                       valid_roi=True)
        roi_list.append(roi)

    return roi_list


@pytest.fixture
def example_roi0():
    rng = np.random.RandomState(64322)
    roi = OphysROI(roi_id=4,
                   x0=10,
                   y0=22,
                   width=7,
                   height=11,
                   valid_roi=True,
                   mask_matrix=rng.randint(0, 2,
                                           (11, 7)).astype(bool))

    return roi


@pytest.fixture(scope='session')
def list_of_roi():
    """
    A list of ExtractROIs
    """
    output = []
    rng = np.random.default_rng(11231)
    for ii in range(10):
        x0 = int(rng.integers(0, 30))
        y0 = int(rng.integers(0, 30))
        width = int(rng.integers(4, 10))
        height = int(rng.integers(4, 10))
        mask = rng.integers(0, 2, size=(height, width)).astype(bool)

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

        if ii % 2 == 0:
            valid_roi = True
        else:
            valid_roi = False
        roi = ExtractROI(x=x0, width=width,
                         y=y0, height=height,
                         valid_roi=valid_roi,
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
