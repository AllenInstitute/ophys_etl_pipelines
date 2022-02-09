import pytest
import numpy as np
import copy
from itertools import product
from ophys_etl.types import OphysROI, ExtractROI
from ophys_etl.utils.rois import (
    extract_roi_to_ophys_roi)
from ophys_etl.utils.roi_plotting_utils import (
    add_roi_contour_to_img,
    add_list_of_roi_contours_to_img)


@pytest.fixture(scope='session')
def extract_roi_list_fixture():

    roi0 = ExtractROI(
                id=0,
                x=1,
                y=1,
                width=4,
                height=5,
                valid=True,
                mask=[[True, True, True, False],
                      [True, True, True, True],
                      [True, True, True, False],
                      [True, True, True, False],
                      [True, True, True, False]])

    roi1 = ExtractROI(
                id=1,
                x=5,
                y=5,
                width=4,
                height=3,
                valid=True,
                mask=[[True, True, True, True],
                      [True, True, True, True],
                      [True, True, True, True]])

    return [roi0, roi1]


@pytest.fixture(scope='session')
def corrupted_extract_roi_list_fixture(
        extract_roi_list_fixture):

    output = []
    for roi in extract_roi_list_fixture:
        new_roi = copy.deepcopy(roi)
        new_roi['valid_roi'] = new_roi.pop('valid')
        new_roi['roi_id'] = new_roi.pop('id')
        new_roi['mask_matrix'] = new_roi.pop('mask')
        output.append(new_roi)
    return output


@pytest.fixture(scope='session')
def ophys_roi_list_fixture(extract_roi_list_fixture):
    return [extract_roi_to_ophys_roi(roi)
            for roi in extract_roi_list_fixture]


@pytest.fixture(scope='session')
def color_map_fixture():
    return {0: (0, 255, 0),
            1: (0, 0, 255)}


@pytest.mark.parametrize('alpha', [0.2, 0.3, 0.4])
def test_add_roi_contour_to_img(alpha):
    img = 100*np.ones((64, 64, 3), dtype=int)

    height = 7
    width = 12

    mask = np.zeros((height, width), dtype=bool)
    mask[1, 5:7] = True
    mask[2, 4:8] = True
    mask[3, 3:9] = True
    mask[4, 2:10] = True
    mask[5, 3:9] = True

    bdry_pixels = set([(1, 5), (1, 6), (2, 4), (2, 7),
                       (3, 3), (3, 8), (4, 2), (4, 9),
                       (5, 3), (5, 4), (5, 5), (5, 6),
                       (5, 7), (5, 8)])

    roi = OphysROI(x0=20, width=width,
                   y0=15, height=height,
                   valid_roi=True, roi_id=0,
                   mask_matrix=mask)

    color = (22, 33, 44)
    img = add_roi_contour_to_img(
                      img,
                      roi,
                      color,
                      alpha)

    for row in range(height):
        for col in range(width):
            for ic in range(3):
                if (row, col) not in bdry_pixels:
                    assert img[15+row, 20+col, ic] == 100
                else:
                    expected = np.round(alpha*color[ic]
                                        + (1.0-alpha)*100).astype(int)
                    assert img[15+row, 20+col, ic] == expected


@pytest.mark.parametrize(
    "roi_list_choice, use_color_map, alpha",
    product((0, 1, 2),
            (True, False),
            (0.5, 0.6)))
def test_add_list_of_roi_contours_to_img(
        extract_roi_list_fixture,
        corrupted_extract_roi_list_fixture,
        ophys_roi_list_fixture,
        color_map_fixture,
        roi_list_choice,
        use_color_map,
        alpha):

    if roi_list_choice == 0:
        roi_list = extract_roi_list_fixture
    elif roi_list_choice == 1:
        roi_list = corrupted_extract_roi_list_fixture
    elif roi_list_choice == 2:
        roi_list = ophys_roi_list_fixture

    if use_color_map:
        color = color_map_fixture
    else:
        color = (0, 125, 125)

    img_value = 55
    img = img_value*np.ones((20, 20, 3), dtype=np.uint8)

    result = add_list_of_roi_contours_to_img(
                img=img,
                roi_list=roi_list,
                color=color,
                alpha=alpha)

    # make sure that input img as not changed
    assert (img == img_value).all()
    assert not np.array_equal(img, result)

    assert result.shape == img.shape
    assert result.dtype == np.uint8

    not_roi_mask = np.ones((20, 20), dtype=bool)

    # check that ROI contour pixels were all set to correct color
    for roi in ophys_roi_list_fixture:
        not_roi_mask[roi.y0:roi.y0+roi.height,
                     roi.x0:roi.x0+roi.width][roi.contour_mask] = False

        if isinstance(color, tuple):
            this_color = color
        else:
            this_color = color[roi.roi_id]
        for ic in range(3):
            expected = np.round(alpha*this_color[ic]
                                + (1.0-alpha)*img_value).astype(np.uint8)
            channel = result[:, :, ic]
            actual_contour = channel[roi.y0:roi.y0+roi.height,
                                     roi.x0:roi.x0+roi.width][roi.contour_mask]
            assert (actual_contour == expected).all()

    # check that pixels not in the ROI contour were all left untouched
    for ic in range(3):
        channel = result[:, :, ic]
        assert (channel[not_roi_mask] == img_value).all()
