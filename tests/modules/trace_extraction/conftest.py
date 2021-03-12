import pytest
import numpy as np

from ophys_etl.modules.trace_extraction.roi_masks import RoiMask


@pytest.fixture
def image_dims():
    return {
        'width': 100,
        'height': 100
    }


@pytest.fixture
def motion_border():
    return [5.0, 5.0, 5.0, 5.0]


@pytest.fixture
def motion_border_dict():
    return {"x0": 5.0, "x1": 5.0, "y0": 5.0, "y1": 5.0}


@pytest.fixture
def roi_mask_list(image_dims, motion_border):
    base_pixels = np.argwhere(np.ones((10, 10)))

    masks = []
    for ii in range(10):
        pixels = base_pixels + ii * 10
        masks.append(RoiMask.create_roi_mask(
            image_dims['width'],
            image_dims['height'],
            motion_border,
            pix_list=pixels,
            label=str(ii),
            mask_group=-1
        ))

    return masks


@pytest.fixture
def roi_list_of_dicts(image_dims, motion_border):
    base_pixels = np.ones((10, 10), dtype=bool).tolist()
    rois = []
    for ii in range(10):
        rois.append({
            "x": ii * 10,
            "y": ii * 10,
            "id": ii,
            "mask": base_pixels})
    return rois
