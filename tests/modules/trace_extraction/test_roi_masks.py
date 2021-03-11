import numpy as np
import pandas as pd
import pytest
from ophys_etl.modules.trace_extraction.roi_masks import (
        RoiMask, NeuropilMask, create_roi_mask_array, validate_mask)


def test_init_by_pixels():
    a = np.array([[0, 0], [1, 1], [1, 0]])

    m = RoiMask.create_roi_mask(2, 2, [0, 0, 0, 0], pix_list=a)

    mp = m.get_mask_plane()

    assert mp[0, 0] == 1
    assert mp[1, 1] == 1
    assert mp[1, 0] == 0
    assert mp[1, 1] == 1

    assert m.x == 0
    assert m.width == 2
    assert m.y == 0
    assert m.height == 2


def test_init_by_pixels_with_border():
    a = np.array([[1, 1], [2, 1]])

    m = RoiMask.create_roi_mask(3, 3, [1, 1, 1, 1], pix_list=a)

    assert m.x == 1
    assert m.width == 2
    assert m.y == 1
    assert m.height == 1
    assert m.overlaps_motion_border is True


def test_init_by_pixels_large():
    a = np.random.random((512, 512))
    a[a > 0.5] = 1

    m = RoiMask.create_roi_mask(
        512, 512, [0, 0, 0, 0], pix_list=np.argwhere(a))

    npx = len(np.where(a)[0])
    assert npx == len(np.where(m.get_mask_plane())[0])


def test_create_neuropil_mask():

    image_width = 100
    image_height = 80

    # border = [image_width-1, 0, image_height-1, 0]
    border = [5, 5, 5, 5]

    roi_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    roi_mask[40:45, 30:35] = 1

    combined_binary_mask = np.zeros((image_height, image_width),
                                    dtype=np.uint8)
    combined_binary_mask[:, 45:] = 1

    roi = RoiMask.create_roi_mask(image_w=image_width,
                                  image_h=image_height,
                                  border=border,
                                  roi_mask=roi_mask)
    obtained = NeuropilMask.create_neuropil_mask(roi, border,
                                                 combined_binary_mask)

    expected_mask = np.zeros((58-27, 45-17), dtype=np.uint8)
    expected_mask[:, :] = 1

    assert np.allclose(expected_mask, obtained.mask)
    assert obtained.x == 17
    assert obtained.y == 27
    assert obtained.width == 28
    assert obtained.height == 31


def test_create_empty_neuropil_mask():
    image_width = 100
    image_height = 80

    # border = [image_width-1, 0, image_height-1, 0]
    border = [5, 5, 5, 5]

    roi_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    roi_mask[40:45, 30:35] = 1

    combined_binary_mask = np.zeros((image_height, image_width),
                                    dtype=np.uint8)
    combined_binary_mask[:, :] = 1

    roi = RoiMask.create_roi_mask(image_w=image_width,
                                  image_h=image_height,
                                  border=border,
                                  roi_mask=roi_mask)
    obtained = NeuropilMask.create_neuropil_mask(roi, border,
                                                 combined_binary_mask)

    assert obtained.mask is None
    assert 'zero_pixels' in obtained.flags


@pytest.fixture
def neuropil_masks(roi_mask_list, motion_border):
    neuropil_masks = []

    mask_array = create_roi_mask_array(roi_mask_list)
    combined_mask = mask_array.max(axis=0)

    for roi_mask in roi_mask_list:
        neuropil_masks.append(NeuropilMask.create_neuropil_mask(
            roi_mask,
            motion_border,
            combined_mask,
            roi_mask.label
        ))
    return neuropil_masks


def test_validate_masks(roi_mask_list, neuropil_masks):
    roi_mask_list.extend(neuropil_masks)
    roi_mask_list[3].mask = np.zeros_like(roi_mask_list[3].mask)
    roi_mask_list[17].mask = np.zeros_like(roi_mask_list[17].mask)

    obtained = []
    for mask in roi_mask_list:
        obtained.extend(validate_mask(mask))

    expected_exclusions = pd.DataFrame({
        'roi_id': ['0', '3', '9', '7'],
        'exclusion_label_name': ['motion_border',
                                 'empty_roi_mask',
                                 'motion_border',
                                 'empty_neuropil_mask']})
    pd.testing.assert_frame_equal(expected_exclusions,
                                  pd.DataFrame(obtained),
                                  check_like=True)
