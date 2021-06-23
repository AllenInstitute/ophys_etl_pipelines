import pytest
import numpy as np
from ophys_etl.modules.segmentation.postprocess_utils.roi_types import (
    SegmentationROI)
from ophys_etl.modules.segmentation.\
    postprocess_utils.roi_time_correlation import (
        get_brightest_pixel,
        sub_video_from_roi,
        correlate_sub_video,
        calculate_merger_metric)


@pytest.fixture
def example_roi0():
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


@pytest.fixture
def example_roi1():
    rng = np.random.RandomState(56423)
    roi = SegmentationROI(roi_id=5,
                          x0=15,
                          y0=26,
                          width=13,
                          height=8,
                          valid_roi=True,
                          flux_value=0.0,
                          mask_matrix=rng.randint(0, 2,
                                                  (8, 13)).astype(bool))

    return roi


def test_get_brightest_pixel(example_roi0):
    ntime = 100
    nrows = 50
    ncols = 50
    rng = np.random.RandomState(771234)
    img_data = rng.randint(11, 355, (nrows, ncols))
    video_data = rng.random_sample((ntime, nrows, ncols))
    sub_video = sub_video_from_roi(example_roi0, video_data)
    brightest_pixel = get_brightest_pixel(example_roi0,
                                          img_data,
                                          sub_video)

    expected = None
    expected_flux = -1.0
    mask = example_roi0.mask_matrix
    for ir in range(example_roi0.height):
        for ic in range(example_roi0.width):
            if not mask[ir, ic]:
                continue
            flux = img_data[example_roi0.y0+ir,
                            example_roi0.x0+ic]
            if flux > expected_flux:
                expected_flux = flux
                expected = video_data[:,
                                      example_roi0.y0+ir,
                                      example_roi0.x0+ic]
    np.testing.assert_array_equal(expected, brightest_pixel)


def test_sub_video_from_roi(example_roi0):
    rng = np.random.RandomState(51433)
    video_data = rng.randint(11, 457, (100, 50, 47))

    sub_video = sub_video_from_roi(example_roi0, video_data)

    npix = example_roi0.mask_matrix.sum()
    expected = np.zeros((100, npix), dtype=int)
    mask = example_roi0.mask_matrix
    i_pix = 0
    for ir in range(example_roi0.height):
        for ic in range(example_roi0.width):
            if not mask[ir, ic]:
                continue
            expected[:, i_pix] = video_data[:,
                                            example_roi0.y0+ir,
                                            example_roi0.x0+ic]
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


@pytest.mark.parametrize('filter_fraction', [0.1, 0.2, 0.3])
def test_calculate_merger_metric(example_roi0, example_roi1, filter_fraction):
    rng = np.random.RandomState(7612)
    nt = 100
    nr = 50
    nc = 50
    video_data = rng.random_sample((nt, nr, nc))
    img_data = rng.random_sample((nr, nc))

    video_lookup = {}
    video_lookup[example_roi0.roi_id] = sub_video_from_roi(example_roi0,
                                                           video_data)

    video_lookup[example_roi1.roi_id] = sub_video_from_roi(example_roi1,
                                                           video_data)

    # brute force get brightest pixel
    brightest_pixel = {}
    for roi in (example_roi0, example_roi1):
        mask = roi.mask_matrix
        pixel = None
        flux = None
        for ir in range(mask.shape[0]):
            for ic in range(mask.shape[1]):
                if not mask[ir, ic]:
                    continue
                if flux is None or img_data[roi.y0+ir, roi.x0+ic] > flux:
                    flux = img_data[roi.y0+ir, roi.x0+ic]
                    pixel = video_data[:, roi.y0+ir, roi.x0+ic]
        brightest_pixel[roi.roi_id] = pixel

    for roi_pair in ((example_roi0, example_roi1),
                     (example_roi1, example_roi0)):
        roi0 = roi_pair[0]
        roi1 = roi_pair[1]
        key_pixel = brightest_pixel[roi0.roi_id]
        corr0 = correlate_sub_video(video_lookup[roi0.roi_id],
                                    key_pixel,
                                    filter_fraction=filter_fraction)
        corr1 = correlate_sub_video(video_lookup[roi1.roi_id],
                                    key_pixel,
                                    filter_fraction=filter_fraction)

        mu = np.mean(corr0)
        std = np.std(corr0, ddof=1)
        z_score = ((corr1-mu)/std)
        expected = np.median(z_score)
        actual = calculate_merger_metric(roi0,
                                         roi1,
                                         video_lookup,
                                         img_data,
                                         filter_fraction=filter_fraction)
        assert np.abs(expected) > 0.0
        assert np.abs((expected-actual)/expected) < 1.0e-10