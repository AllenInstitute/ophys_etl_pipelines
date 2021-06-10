import pytest
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.postprocess_utils.roi_merging import (
    merge_rois,
    do_rois_abut,
    correlate_sub_videos,
    make_cdf,
    find_merger_candidates)

@pytest.fixture
def example_roi_list():
    rng = np.random.RandomState(6412439)
    roi_list = []
    for ii in range(30):
        x0 = rng.randint(0, 25)
        y0 = rng.randint(0, 25)
        height = rng.randint(3, 7)
        width = rng.randint(3, 7)
        mask = rng.randint(0, 2, (height, width)).astype(bool)
        roi = OphysROI(x0=x0, y0=y0,
                       height=height, width=width,
                       mask_matrix=mask,
                       roi_id=ii,
                       valid_roi=True)
        roi_list.append(roi)

    return roi_list

def test_merge_rois():

    x0 = 11
    y0 = 22
    height = 5
    width = 6
    mask = np.zeros((height, width), dtype=bool)
    mask[4, 5] = True
    mask[3, 5] = True
    mask[0, 0] = True
    mask[1, 4] = True
    roi0 = OphysROI(x0=x0,
                    width=width,
                    y0=y0,
                    height=height,
                    mask_matrix=mask,
                    roi_id=0,
                    valid_roi=True)

    y0=19
    x0=16
    height=6
    width=4
    mask = np.zeros((height, width), dtype=bool)
    mask[5, 0] = True
    mask[5, 1] = True
    mask[5, 2] = True
    mask[3, 3] = True

    roi1 = OphysROI(x0=x0,
                    y0=y0,
                    width=width,
                    height=height,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    merged_roi = merge_rois(roi0, roi1, 2)
    assert merged_roi.roi_id == 2
    assert merged_roi.x0 == 11
    assert merged_roi.y0 == 19
    assert merged_roi.width == 9
    assert merged_roi.height == 8

    # make sure all pixels that should be marked
    # True are
    true_pix = set()
    new_mask = merged_roi.mask_matrix
    for roi in (roi0, roi1):
        x0 = roi.x0
        y0 = roi.y0
        mask = roi.mask_matrix
        for ir in range(roi.height):
            for ic in range(roi.width):
                if not mask[ir, ic]:
                    continue
                row = ir+y0-merged_roi.y0
                col = ic+x0-merged_roi.x0
                assert new_mask[row, col]
                true_pix.add((row, col))
    # make sure no extraneous pixels were marked True
    assert len(true_pix) == new_mask.sum()


def test_roi_abut():

    height = 6
    width = 7
    mask = np.zeros((height, width), dtype=bool)
    mask[1:5, 1:6] = True

    # overlapping
    roi0 = OphysROI(x0=22,
                    y0=44,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=0,
                    valid_roi=True) 

    roi1 = OphysROI(x0=23,
                    y0=46,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert do_rois_abut(roi0, roi1)

    # just touching
    roi1 = OphysROI(x0=26,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert do_rois_abut(roi0, roi1)

    roi1 = OphysROI(x0=27,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert not do_rois_abut(roi0, roi1)

    # they are, however, just diagonally 1 pixel away
    # from each other
    assert do_rois_abut(roi0, roi1, dpix=np.sqrt(2))

    # gap of one pixel
    assert do_rois_abut(roi0, roi1, dpix=2)

    roi1 = OphysROI(x0=28,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert not do_rois_abut(roi0, roi1, dpix=2)


def test_correlate_sub_videos():
    """
    compare to brute force
    """
    rng = np.random.RandomState(11823)
    nt = 2000
    npix0 = 20
    npix1 = 10
    filter_fraction = 0.2
    video0 = rng.random_sample((nt, npix0))
    video1 = rng.random_sample((nt, npix1))
    corr = correlate_sub_videos(video0, video1, filter_fraction)

    for ipix0 in range(npix0):
        trace0 = video0[:,ipix0]
        th = np.quantile(trace0, 1.0-filter_fraction)
        mask = (trace0>th)
        trace0 = trace0[mask]
        mu0 = np.mean(trace0)
        var0 = np.mean((trace0-mu0)**2)
        for ipix1 in range(npix1):
            trace1 = video1[:, ipix1]
            trace1 = trace1[mask]
            mu1 = np.mean(trace1)
            var1 = np.mean((trace1-mu1)**2)
            val = np.mean((trace0-mu0)*(trace1-mu1))
            val = val/np.sqrt(var1*var0)
            assert np.abs((val-corr[ipix0,ipix1])/val)<1.0e-10


def test_cdf():
    rng = np.random.RandomState(221144)
    data_set_list = [rng.random_sample(200)*5.0,
                     rng.normal(5.0, 2.0, size=300)]
    for data_set in data_set_list:
        (bins,
         cdf) = make_cdf(data_set)
        test = rng.random_sample(50)*5.2
        interped_cdf = np.interp(test, bins, cdf)
        for xx, yy in zip(test, interped_cdf):
            ct = (data_set<xx).sum()
            assert np.abs(yy-ct/len(data_set)) < 1.0e-2


@pytest.mark.parametrize("dpix",[np.sqrt(2), 4, 5])
def test_find_merger_candidates(dpix, example_roi_list):
    true_matches = set()
    has_been_matched = set()
    for i0 in range(len(example_roi_list)):
        roi0 = example_roi_list[i0]
        for i1 in range(i0+1, len(example_roi_list), 1):
            roi1 = example_roi_list[i1]
            if do_rois_abut(roi0, roi1, dpix=dpix):
                true_matches.add((roi0.roi_id, roi1.roi_id))
                has_been_matched.add(roi0.roi_id)
                has_been_matched.add(roi1.roi_id)

    assert len(has_been_matched) > 0
    assert len(has_been_matched) < len(example_roi_list)
    expected = set(true_matches)

    for n in (3, 5):
        matches = find_merger_candidates(example_roi_list,
                                         dpix,
                                         5)
        matches = set(matches)
        assert matches == expected
