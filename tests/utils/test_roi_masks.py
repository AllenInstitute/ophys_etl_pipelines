import numpy as np
import pandas as pd
import pytest
from ophys_etl.utils.roi_masks import (RoiMask, NeuropilMask,
                                       create_roi_mask_array, validate_mask)


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


@pytest.mark.parametrize(
    "pix_list, expected_x,"
    "expected_width, expected_y,"
    "expected_height, expected_flag",
    [
        # Crosses the left border
        (np.array([[0, 3], [1, 3]]), 0, 2, 3, 1, True),
        # Crosses the right border
        (np.array([[6, 3], [5, 3]]), 5, 2, 3, 1, True),
        # Crosses the top border
        (np.array([[3, 0], [3, 1]]), 3, 1, 0, 2, True),
        # Crosses the bottom border
        (np.array([[3, 6], [3, 5]]), 3, 1, 5, 2, True),
        # Touches the left border
        (np.array([[2, 3], [3, 3]]), 2, 2, 3, 1, False),
        # Touches the right border
        (np.array([[4, 3], [5, 3]]), 4, 2, 3, 1, False),
        # Touches the top border
        (np.array([[3, 2], [3, 3]]), 3, 1, 2, 2, False),
        # Touches the bottom border
        (np.array([[3, 4], [3, 5]]), 3, 1, 4, 2, False),
        # Neither touches nor crosses any border
        (np.array([[3, 3], [4, 3]]), 3, 2, 3, 1, False),
    ],
)
def test_init_by_pixels_with_border(
    pix_list, expected_x, expected_width, expected_y, expected_height, expected_flag # noqa 501
):
    m = RoiMask.create_roi_mask(7, 7, [2, 2, 2, 2], pix_list)
    assert m.x == expected_x
    assert m.width == expected_width
    assert m.y == expected_y
    assert m.height == expected_height
    assert m.overlaps_motion_border is expected_flag


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


# fixtures from tests/conftest.py
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


# fixtures from tests/conftest.py
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
