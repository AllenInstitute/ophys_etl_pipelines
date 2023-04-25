import pytest
import numpy as np


# fixtures from tests/conftest.py
@pytest.fixture
def roi_list_of_dicts(image_dims, motion_border):
    base_pixels = np.ones((10, 10), dtype=bool).tolist()
    rois = []
    for ii in range(10):
        rois.append(
            {"x": ii * 10, "y": ii * 10, "id": ii, "mask": base_pixels}
        )
    return rois
