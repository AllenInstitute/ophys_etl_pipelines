import pytest
import numpy as np
from ophys_etl.modules.segmentation.postprocess_utils.roi_types import (
    SegmentationROI)
from ophys_etl.modules.segmentation.\
    postprocess_utils.roi_time_correlation import (
        get_brightest_pixel,
        sub_video_from_roi,
        correlate_sub_video)


@pytest.fixture
def example_roi():
    rng = np.random.RandomState(64322)
    roi = SegmentationROI(roi_id=4,
                          x0=10,
                          y0=22,
                          width=7,
                          height=11,
                          valid_roi=True,
                          flux_value=0.0,
                          mask_matrix=rng.randint(0, 2,
                                                  (11, 7)).astype(bool))

    return roi


def test_get_brightest_pixel(example_roi):
    ntime = 100
    nrows = 50
    ncols = 50
    rng = np.random.RandomState(771234)
    img_data = rng.randint(11, 355, (nrows, ncols))
    video_data = rng.random_sample((ntime, nrows, ncols))
    brightest_pixel = get_brightest_pixel(example_roi,
                                          img_data,
                                          video_data)

    expected = None
    expected_flux = -1.0
    mask = example_roi.mask_matrix
    for ir in range(example_roi.height):
        for ic in range(example_roi.width):
            if not mask[ir, ic]:
                continue
            flux = img_data[example_roi.y0+ir,
                            example_roi.x0+ic]
            if flux > expected_flux:
                expected_flux = flux
                expected = video_data[:,
                                      example_roi.y0+ir,
                                      example_roi.x0+ic]
    np.testing.assert_array_equal(expected, brightest_pixel)


def test_sub_video_from_roi(example_roi):
    rng = np.random.RandomState(51433)
    video_data = rng.randint(11, 457, (100, 50, 47))

    sub_video = sub_video_from_roi(example_roi, video_data)

    npix = example_roi.mask_matrix.sum()
    expected = np.zeros((100, npix), dtype=int)
    mask = example_roi.mask_matrix
    i_pix = 0
    for ir in range(example_roi.height):
        for ic in range(example_roi.width):
            if not mask[ir, ic]:
                continue
            expected[:, i_pix] = video_data[:,
                                            example_roi.y0+ir,
                                            example_roi.x0+ic]
            i_pix += 1
    np.testing.assert_array_equal(expected, sub_video)


@pytest.mark.parametrize('filter_fraction', [0.1, 0.2, 0.3])
def test_correlate_sub_video(filter_fraction):
    ntime = 137
    npixels = 45
    rng = np.random.RandomState(8234)
    video_data = rng.random_sample((ntime, npixels))
    key_pixel = rng.random_sample(ntime)

    corr = correlate_sub_video(video_data,
                               key_pixel,
                               filter_fraction=filter_fraction)

    th = np.quantile(key_pixel, 1.0-filter_fraction)
    mask = (key_pixel >= th)
    key_pixel = key_pixel[mask]
    mu = np.mean(key_pixel)
    var = np.var(key_pixel)
    for i_pix in range(npixels):
        pixel = video_data[:, i_pix]
        pixel = pixel[mask]
        mu_p = np.mean(pixel)
        var_p = np.var(pixel)
        expected = np.mean((key_pixel-mu)*(pixel-mu_p))/np.sqrt(var*var_p)
        assert np.abs((expected-corr[i_pix])/expected) < 1.0e-6
