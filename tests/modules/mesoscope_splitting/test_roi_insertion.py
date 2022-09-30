# methods in this module will test the full_field_utils
# that specifically insert ROI images into the stitched
# full field images

import pytest
import tempfile
import pathlib
import tifffile
import numpy as np
from skimage.transform import resize as skimage_resize
from unittest.mock import Mock, patch

from ophys_etl.utils.array_utils import (
    normalize_array)

from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)

from ophys_etl.modules.mesoscope_splitting.tiff_splitter import (
    AvgImageTiffSplitter)

from ophys_etl.modules.mesoscope_splitting.full_field_utils import (
    _insert_rois_into_surface_img)


def _create_avg_surface_tiff(
        n_rois: int,
        pixel_size: float,
        origin_x: float,
        origin_y: float,
        output_dir: pathlib.Path):
    """
    Returns 12 x 12 ROIs where each pixel in the ROI
    corresponds to pixel_size physical units

    The ROIs are centered at

    centerX = origin_x + (i_roi+1)*24
    centerY = origin_y + (i_roi+1)*36

    origin_x, origin_y denote the origin of the background
    image in physical units

    Return
    ------
    path to tiff file

    list of average images

    metadata to be passed to mock
    """

    nrows = 12
    ncols = 12

    rng = np.random.default_rng(562132)

    tiff_path = pathlib.Path(
           tempfile.mkstemp(dir=output_dir, suffix='.tiff')[1])

    metadata = [dict(), dict()]
    avg_images = []

    n_pages_per_roi = 5
    data = np.zeros((n_pages_per_roi*n_rois,
                     nrows,
                     ncols))

    for i_roi in range(n_rois):
        this_data = rng.integers(
                        i_roi*100,
                        (i_roi+1)*100,
                        (n_pages_per_roi, nrows, ncols)).astype(np.uint16)
        this_avg = np.mean(this_data, axis=0)
        avg_images.append(this_avg)
        for i_page in range(i_roi, n_pages_per_roi*n_rois, n_rois):
            data[i_page, :, :] = this_data[i_page//n_rois, :, :]

    tifffile.imsave(tiff_path, data)

    metadata[0][
        'SI.hStackManager.zsAllActuators'] = [[i_roi, 0]
                                              for i_roi in range(n_rois)]

    metadata[0][
        'SI.hChannels.channelSave'] = 1

    roi_metadata = []
    for i_roi in range(n_rois):
        this_roi = {'zs': i_roi,
                    'scanfields': {
                        'pixelResolutionXY': [ncols, nrows],
                        'sizeXY': [ncols*pixel_size, nrows*pixel_size],
                        'centerXY': [origin_x + 24*(i_roi+1)*pixel_size,
                                     origin_y + 36*(i_roi+1)*pixel_size]}}
        roi_metadata.append(this_roi)

    metadata[1]['RoiGroups'] = {
        'imagingRoiGroup': {
            'rois': roi_metadata}}

    return (tiff_path,
            avg_images,
            metadata)


@pytest.mark.parametrize(
        "n_rois, bckgd_pixel, roi_pixel",
        [(3, 1.1, 2.2),
         (4, 3.2, 1.6)])
def test_insertion_worker(
        tmpdir_factory,
        helper_functions,
        n_rois,
        bckgd_pixel,
        roi_pixel):
    """
    Test _insert_rois_into_surface_img

    n_rois -- number of ROIs
    bckgd_pixel -- physical size of one pixel in the background image
    roi_pixel -- physical size of one pixel in the ROIs
    """
    origin_x = 4.3
    origin_y = 3.2
    bckgd_img = np.NaN*np.ones((512, 512), dtype=float)

    tmpdir = pathlib.Path(
            tmpdir_factory.mktemp('insertion_worker'))

    (tiff_path,
     expected_avg,
     avg_metadata) = _create_avg_surface_tiff(
                         n_rois=n_rois,
                         pixel_size=roi_pixel,
                         origin_x=origin_x,
                         origin_y=origin_y,
                         output_dir=tmpdir)

    with patch("tifffile.read_scanimage_metadata",
               new=Mock(return_value=avg_metadata)):
        avg_splitter = AvgImageTiffSplitter(tiff_path)
    for i_roi in range(n_rois):
        actual = avg_splitter.get_avg_img(i_roi=i_roi, z_value=None)
        np.testing.assert_allclose(actual, expected_avg[i_roi])

    # create metadata dict for the background image
    sizexy = [512*bckgd_pixel, 512*bckgd_pixel]
    bckgd_metadata = [dict(), dict()]
    bckgd_metadata[1]['RoiGroups'] = {
        'imagingRoiGroup': {
            'rois': [{'scanfields':
                      {'pixelResolutionXY': [512, 512],
                       'sizeXY': sizexy,
                       'centerXY': [origin_x+0.5*sizexy[0],
                                    origin_y+0.5*sizexy[1]]}}]}}

    nonsense_path = pathlib.Path(
            tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    with patch("tifffile.read_scanimage_metadata",
               new=Mock(return_value=bckgd_metadata)):
        ff_metadata = ScanImageMetadata(nonsense_path)

    img = _insert_rois_into_surface_img(
            full_field_img=bckgd_img,
            full_field_metadata=ff_metadata,
            avg_image_splitter=avg_splitter)

    # mask out expected ROI pixels so that we can verify that
    # non-ROI pixels are still NaN after insertion
    mask_missing = np.ones((512, 512), dtype=bool)

    # make sure that ROIs were correctly inserted into the
    # image
    for i_roi in range(n_rois):
        expected = expected_avg[i_roi]
        new_dim = np.round(12*roi_pixel/bckgd_pixel).astype(int)
        expected = skimage_resize(expected, (new_dim, new_dim))
        expected = normalize_array(expected, dtype=np.uint16)

        roi_col0 = 24*(i_roi+1)-6
        roi_col0 = np.round(roi_col0*roi_pixel/bckgd_pixel).astype(int)
        roi_row0 = 36*(i_roi+1)-6
        roi_row0 = np.round(roi_row0*roi_pixel/bckgd_pixel).astype(int)

        actual = img[roi_row0: roi_row0+expected.shape[0],
                     roi_col0: roi_col0+expected.shape[1]]

        mask_missing[roi_row0: roi_row0+expected.shape[0],
                     roi_col0: roi_col0+expected.shape[1]] = False

        np.testing.assert_allclose(actual, expected)

    # verify that all of the pixels that should not be in the ROIs are
    # still NaNs
    assert np.all(np.isnan(img[mask_missing]))

    helper_functions.clean_up_dir(tmpdir)
