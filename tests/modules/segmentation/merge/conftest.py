import pytest
import numpy as np
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
