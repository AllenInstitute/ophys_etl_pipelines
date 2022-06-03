from ophys_etl.types import OphysROI
import numpy as np


def test_roi_instantiation(
        ophys_plane_data_fixture):
    schema_dict = ophys_plane_data_fixture
    ct = 0
    for meta_pair in schema_dict['coupled_planes']:
        pair = meta_pair['planes']
        for plane in pair:
            for roi_args in plane['rois']:
                _ = OphysROI.from_schema_dict(roi_args)
                ct += 1
    assert ct == 8


def test_roi_global_pixel_set():
    width = 7
    height = 5
    mask = np.zeros((height, width), dtype=bool)
    mask[2, 4] = True
    mask[3, 6] = True
    roi = OphysROI(roi_id=1,
                   x0=100,
                   y0=200,
                   width=width,
                   height=height,
                   valid_roi=True,
                   mask_matrix=mask)
    assert roi.global_pixel_set == set([(202, 104), (203, 106)])


def test_get_bound_box_cutout():
    """Test retrieval of a cutout of the roi bounding box.
    """
    rng = np.random.default_rng(1234)
    image = rng.integers(100, size=(512, 512), dtype='int16')
    width = 7
    height = 5
    x0 = 100
    y0 = 200
    mask = np.zeros((height, width), dtype=bool)
    mask[2, 4] = True
    mask[3, 6] = True
    roi = OphysROI(roi_id=1,
                   x0=x0,
                   y0=y0,
                   width=width,
                   height=height,
                   valid_roi=True,
                   mask_matrix=mask)

    cutout = roi.get_bounding_box_cutout(image)
    np.testing.assert_array_equal(cutout,
                                  image[y0:y0 + height, x0:x0 + width])


def test_get_centered_cutout():
    """Test getting a cutout of arbitrary size centered on the roi
    bounding box center.
    """
    rng = np.random.default_rng(1234)
    image = rng.integers(100, size=(512, 512), dtype='int16')
    width = 10
    height = 10
    x0 = 100
    y0 = 200
    mask = np.zeros((height, width), dtype=bool)
    mask[2, 4] = True
    mask[3, 6] = True
    roi = OphysROI(roi_id=1,
                   x0=x0,
                   y0=y0,
                   width=width,
                   height=height,
                   valid_roi=True,
                   mask_matrix=mask)

    cutout_size = 128
    cutout = roi.get_centered_cutout(image, cutout_size, cutout_size)
    np.testing.assert_array_equal(
        cutout,
        image[roi.bounding_box_center_y - cutout_size // 2:
              roi.bounding_box_center_y + cutout_size // 2,
              roi.bounding_box_center_x - cutout_size // 2:
              roi.bounding_box_center_x + cutout_size // 2])
