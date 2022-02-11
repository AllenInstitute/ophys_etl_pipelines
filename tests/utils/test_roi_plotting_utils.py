import pytest
import numpy as np
import copy
from itertools import product
from ophys_etl.types import OphysROI, ExtractROI
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.utils.rois import (
    extract_roi_to_ophys_roi)
from ophys_etl.utils.roi_plotting_utils import (
    add_roi_contour_to_img,
    add_list_of_roi_contours_to_img,
    plot_rois_over_img)


@pytest.fixture(scope='session')
def extract_roi_list_fixture():
    """A list of ExtractROIs"""
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
    """A list of ExtractROIs with the wrong keys ('valid_roi' instead
    of 'valid', etc.)"""
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
    """A list of OphysROIs"""
    return [extract_roi_to_ophys_roi(roi)
            for roi in extract_roi_list_fixture]


@pytest.fixture(scope='session')
def color_map_fixture():
    """an example color map"""
    return {0: (11, 255, 56),
            1: (0, 0, 255)}


@pytest.mark.parametrize('alpha', [0.2, 0.3, 0.4])
def test_add_roi_contour_to_img(alpha):
    """Test that add_roi_contour_to_img creates an image with
    the expected contours of the expected colors"""
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
    """
    Test taht add_list_of_roi_contours_to_img adds contours
    of the expected colors to an image
    """

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


@pytest.mark.parametrize(
    "roi_list_choice, use_color_map, alpha, use_rgb, use_float, blank_image",
    product((0, 1, 2),
            (True, False),
            (0.5, 0.6),
            (True, False),
            (True, False),
            (True, False)))
def test_plot_rois_over_img(
        extract_roi_list_fixture,
        corrupted_extract_roi_list_fixture,
        ophys_roi_list_fixture,
        color_map_fixture,
        roi_list_choice,
        use_color_map,
        alpha,
        use_rgb,
        use_float,
        blank_image):
    """
    Test that plot_rois_over_img produces an image with the expected
    contours drawn at the expected place in the expected colors
    """
    rng = np.random.default_rng(7612322)

    # choose a list of ROIs
    if roi_list_choice == 0:
        roi_list = extract_roi_list_fixture
    elif roi_list_choice == 1:
        roi_list = corrupted_extract_roi_list_fixture
    elif roi_list_choice == 2:
        roi_list = ophys_roi_list_fixture

    # either unique colors, or one color for all ROIs
    if use_color_map:
        color = color_map_fixture
    else:
        color = (0, 255, 0)

    # create an input image
    if use_float:
        if blank_image:
            img = 2000.0*np.ones((20, 20), dtype=float)
            if use_rgb:
                img = np.stack([img, img, img]).transpose(1, 2, 0)
        else:
            if use_rgb:
                img = rng.random((20, 20, 3))*2111.0
            else:
                img = rng.random((20, 20))*2111.0
    else:
        if blank_image:
            img = 111*np.zeros((20, 20), dtype=np.uint8)
            if use_rgb:
                img = np.stack([img, img, img]).transpose(1, 2, 0)
        else:
            if use_rgb:
                img = rng.integers(0, np.iinfo(np.uint8).max,
                                   (20, 20, 3), dtype=np.uint8)
            else:
                img = rng.integers(0, np.iinfo(np.uint8).max,
                                   (20, 20), dtype=np.uint8)

    result = plot_rois_over_img(
                img=img,
                roi_list=roi_list,
                color=color,
                alpha=alpha)

    # expected_image is what we would expect the image to be
    # without any ROIs plotted over it
    if blank_image:
        expected_image = np.zeros((20, 20, 3), dtype=np.uint8)
    else:
        expected_image = normalize_array(
                              array=img,
                              lower_cutoff=img.min(),
                              upper_cutoff=img.max())

        if len(expected_image.shape) < 3:
            expected_image = np.stack([expected_image,
                                       expected_image,
                                       expected_image]).transpose(1, 2, 0)

    # make sure that input img as not changed
    assert not np.array_equal(img, result)
    assert not np.array_equal(expected_image, result)

    # check the shape and dtype of the output
    assert result.shape == (img.shape[0], img.shape[1], 3)
    assert result.dtype == np.uint8

    # keep track of pixels that are not part of ROI contour
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

            expected_channel = expected_image[roi.y0:roi.y0+roi.height,
                                              roi.x0:roi.x0+roi.width, ic]
            expected_channel = expected_channel[roi.contour_mask].flatten()
            expected_channel = np.round(
                                alpha*this_color[ic]
                                + (1.0-alpha)*expected_channel)
            expected_channel = expected_channel.astype(np.uint8)

            channel = result[:, :, ic]
            actual_channel = channel[roi.y0:roi.y0+roi.height,
                                     roi.x0:roi.x0+roi.width][roi.contour_mask]
            actual_channel = actual_channel.flatten()
            np.testing.assert_array_equal(actual_channel,
                                          expected_channel)

    # check that pixels not in the ROI contour were all left untouched
    for ic in range(3):
        channel = result[:, :, ic]
        expected_channel = expected_image[:, :, ic]
        np.testing.assert_array_equal(channel[not_roi_mask],
                                      expected_channel[not_roi_mask])
