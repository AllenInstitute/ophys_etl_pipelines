import pytest
import numpy as np
from ophys_etl.modules.segmentation.postprocess_utils.roi_types import (
    SegmentationROI)
from ophys_etl.modules.segmentation.\
    postprocess_utils.roi_time_correlation import (
        get_brightest_pixel,
        get_brightest_pixel_parallel,
        sub_video_from_roi,
        correlate_sub_video,
        calculate_merger_metric,
        _self_correlate,
        _correlate_batch,
        _wgts_to_series)


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


@pytest.fixture
def example_video():
    rng = np.random.RandomState(1172312)
    data = rng.random_sample((100, 50, 50))
    return data


@pytest.mark.parametrize('i_pixel, filter_fraction',
                         [(22, 0.2), (22, 0.4),
                          (13, 0.2), (13, 0.4)])
def test_self_correlate(example_video, i_pixel, filter_fraction):

    example_video = example_video.reshape(example_video.shape[0], -1)
    this_pixel = example_video[:, i_pixel]
    th = np.quantile(this_pixel, 1.0-filter_fraction)
    this_mask = (this_pixel >= th)
    expected = np.zeros(example_video.shape[1], dtype=float)
    for i_other in range(example_video.shape[1]):
        other_pixel = example_video[:, i_other]
        th = np.quantile(other_pixel, 1.0-filter_fraction)
        other_mask = (other_pixel >= th)
        mask = np.logical_or(this_mask, other_mask)
        p0 = this_pixel[mask]
        p1 = other_pixel[mask]

        denom = np.sqrt(np.var(p0, ddof=1)*np.var(p1, ddof=1))
        num = np.mean((p0-np.mean(p0))*(p1-np.mean(p1)))

        expected[i_other] = num/denom

    np.testing.assert_array_equal(expected.sum(),
                                  _self_correlate(example_video,
                                                  i_pixel,
                                                  filter_fraction))

@pytest.mark.parametrize('filter_fraction',
                         [0.2, 0.3, 0.4])
def test_correlate_batch(example_video, filter_fraction):
    example_video = example_video.reshape(example_video.shape[0], -1)
    pixel_list = np.array([4, 6, 110, 2000, 300])
    output_dict = {}
    _correlate_batch(pixel_list,
                     example_video,
                     output_dict,
                     filter_fraction=filter_fraction)
    for ipix in pixel_list:
        expected = _self_correlate(example_video,
                                   ipix,
                                   filter_fraction=filter_fraction)
        assert np.abs(expected-output_dict[ipix]) < 1.0e-6


def test_wgts_to_series():
    rng = np.random.RandomState(182312)

    # test case with only one pixel
    sub_video = rng.random_sample((100, 1))
    wgts = np.array([22.1])
    result = _wgts_to_series(sub_video, wgts)
    np.testing.assert_array_equal(result, sub_video[:,0])

    sub_video = rng.random_sample((100, 20))

    # test case where all weights are the same
    wgts = 22.1*np.ones(20, dtype=float)
    result = _wgts_to_series(sub_video, wgts)
    np.testing.assert_allclose(result,
                               np.mean(sub_video, axis=1),
                               atol=1.0e-10,
                               rtol=1.0e-10)

    # test case where weights above the median are uniform
    # (i.e. test that weights below the median still get
    # masked out, even after weights are converged to ones)
    wgts = 22.1*np.ones(20)
    wgts[5] = 3.0
    wgts[11] = 3.0
    wgts[13] = 3.0
    result = _wgts_to_series(sub_video, wgts)
    mask = np.ones(20, dtype=bool)
    mask[5] = False
    mask[11] = False
    mask[13] = False
    np.testing.assert_allclose(result,
                               np.mean(sub_video[:, mask], axis=1),
                               atol=1.0e-10,
                               rtol=1.0e-10)

    # test non-uniform weights
    wgts = rng.random_sample(20)
    med = np.median(wgts)
    norm = np.max(wgts-med)
    mask = (wgts>med)
    masked_wgts = (wgts[mask]-med)/norm
    masked_sub_video = sub_video[:, mask]
    expected = np.zeros(sub_video.shape[0], dtype=float)
    for ii in range(len(masked_wgts)):
        expected += masked_sub_video[:, ii]*masked_wgts[ii]
    expected = expected/(masked_wgts.sum())
    actual = _wgts_to_series(sub_video, wgts)
    np.testing.assert_allclose(expected,
                               actual,
                               atol=1.0e-10,
                               rtol=1.0e-10)


@pytest.mark.parametrize('filter_fraction',
                         [0.1, 0.2, 0.3, 0.4])
def test_get_brightest_pixel(filter_fraction):
    rng = np.random.RandomState(7123412)
    sub_video = rng.random_sample((100, 20))
    wgts = np.zeros(20, dtype=float)
    for ipix in range(20):
        wgts[ipix] = _self_correlate(sub_video,
                                     ipix,
                                     filter_fraction=filter_fraction)
    assert len(np.unique(wgts)) == len(wgts)
    expected = _wgts_to_series(sub_video, wgts)
    actual = get_brightest_pixel(sub_video,
                                 filter_fraction=filter_fraction)
    np.testing.assert_allclose(expected,
                               actual,
                               rtol=1.0e-10,
                               atol=1.0e-10)

@pytest.mark.parametrize('filter_fraction, n_processors',
                         [(0.2, 3), (0.2, 5),
                          (0.3, 3), (0.3, 5)])
def test_get_brightest_pixel_parallel(
        filter_fraction,
        n_processors):
    rng = np.random.RandomState(65423)
    sub_video = rng.random_sample((100, 91))
    expected = get_brightest_pixel(sub_video,
                                   filter_fraction=filter_fraction)
    actual = get_brightest_pixel_parallel(
                 sub_video,
                 filter_fraction=filter_fraction)

    np.testing.assert_allclose(expected,
                               actual,
                               atol=1.0e-10,
                               rtol=1.0e-10)


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
def test_calculate_merger_metric(filter_fraction):
    rng = np.random.RandomState(123412)
    sub_video = rng.random_sample((100, 20))
    key_pixel = rng.random_sample(100)
    distribution_params = (rng.random_sample(), rng.random_sample())

    actual = calculate_merger_metric(
                    distribution_params,
                    key_pixel,
                    sub_video,
                    filter_fraction=filter_fraction)

    corr = correlate_sub_video(sub_video,
                               key_pixel,
                               filter_fraction=filter_fraction)
    z_score = (corr-distribution_params[0])/distribution_params[1]
    expected = np.median(z_score)
    assert np.abs((actual-expected)/expected) < 1.0e-10
