import pytest
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.merge.roi_utils import (
    do_rois_abut)

from ophys_etl.modules.segmentation.merge.roi_merging import (
    find_merger_candidates,
    do_roi_merger)


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


@pytest.fixture
def example_movie_data():
    rng = np.random.RandomState(871123)
    data = np.zeros((100, 60, 60), dtype=float)
    data[:, 30:, 30:] = rng.normal(15.0, 7.0, size=(100, 30, 30))
    data[:, :30, :30] = rng.random_sample((100, 30, 30))*17.0
    return data


@pytest.fixture
def whole_dataset():
    """
    Create a video with a bunch of neighboring, correlated ROIs
    """
    rng = np.random.RandomState(1723133)
    nrows = 100
    ncols = 100
    ntime = 700
    video = rng.randint(0, 100, (ntime, nrows, ncols))

    roi_list = []
    roi_id = 0
    for ii in range(4):
        x0 = rng.randint(14, 61)
        y0 = rng.randint(14, 61)
        height = rng.randint(12, 18)
        width = rng.randint(12, 18)

        mask = np.zeros((height, width)).astype(bool)
        mask[1:-1, 1:-1] = True

        freq = rng.randint(50, 400)
        time_series = np.sin(np.arange(ntime).astype(float)/freq)
        time_series = np.round(time_series).astype(int)
        for ir in range(height):
            for ic in range(width):
                if mask[ir, ic]:
                    video[:, y0+ir, x0+ic] += time_series

        for r0 in range(0, height, height//2):
            for c0 in range(0, width, width//2):
                roi_id += 1
                this_mask = mask[r0:r0+height//2, c0:c0+width//2]
                if this_mask.sum() == 0:
                    continue
                roi = OphysROI(x0=x0+c0, y0=y0+r0,
                               height=this_mask.shape[0],
                               width=this_mask.shape[1],
                               mask_matrix=this_mask,
                               roi_id=roi_id,
                               valid_roi=True)
                roi_list.append(roi)

    return {'video': video, 'roi_list': roi_list}



@pytest.mark.parametrize("dpix", [np.sqrt(2), 4, 5])
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
                                         n_processors=5)
        matches = set(matches)
        assert matches == expected


@pytest.mark.parametrize("dpix", [np.sqrt(2), 4, 5])
def test_find_merger_candidates_with_ignore(dpix, example_roi_list):
    full_matches = set()
    has_been_matched = set()
    for i0 in range(len(example_roi_list)):
        roi0 = example_roi_list[i0]
        for i1 in range(i0+1, len(example_roi_list), 1):
            roi1 = example_roi_list[i1]
            if do_rois_abut(roi0, roi1, dpix=dpix):
                full_matches.add((roi0.roi_id, roi1.roi_id))
                has_been_matched.add(roi0.roi_id)
                has_been_matched.add(roi1.roi_id)

    assert len(has_been_matched) > 0
    assert len(has_been_matched) < len(example_roi_list)
    full_matches = list(full_matches)
    full_matches.sort()

    rois_to_ignore = set()
    rois_to_ignore.add(full_matches[0][0])
    rois_to_ignore.add(full_matches[0][1])
    rois_to_ignore.add(full_matches[10][0])
    rois_to_ignore.add(full_matches[10][1])

    for n in (3, 5):
        matches = find_merger_candidates(example_roi_list,
                                         dpix,
                                         rois_to_ignore=rois_to_ignore,
                                         n_processors=5)
        assert len(matches) > 0
        assert len(matches) < len(full_matches)
        matches = set(matches)
        for m in full_matches:
            if m[0] in rois_to_ignore and m[1] in rois_to_ignore:
                assert m not in matches
            else:
                assert m in matches


def test_do_roi_merger(whole_dataset):
    """
    smoke test for do_roi_merger
    """
    img_data = np.mean(whole_dataset['video'], axis=0)
    assert img_data.shape == whole_dataset['video'].shape[1:]
    new_roi_list = do_roi_merger(whole_dataset['roi_list'],
                                 whole_dataset['video'],
                                 3,
                                 2.0,
                                 filter_fraction=0.2)

    # test that some mergers were performed
    assert len(new_roi_list) > 0
    assert len(new_roi_list) < len(whole_dataset['roi_list'])

    # check that pixels were conserved
    input_pixels = set()
    for roi in whole_dataset['roi_list']:
        mask = roi.mask_matrix
        for ir in range(roi.height):
            for ic in range(roi.width):
                if mask[ir, ic]:
                    pix = (roi.y0+ir, roi.x0+ic)
                    input_pixels.add(pix)

    output_pixels = set()
    for roi in new_roi_list:
        mask = roi.mask_matrix
        for ir in range(roi.height):
            for ic in range(roi.width):
                if mask[ir, ic]:
                    pix = (roi.y0+ir, roi.x0+ic)
                    output_pixels.add(pix)

    assert output_pixels == input_pixels
