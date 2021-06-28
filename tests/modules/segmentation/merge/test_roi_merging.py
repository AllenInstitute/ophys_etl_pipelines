import pytest
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.merge.roi_merging import (
    merge_rois,
    do_rois_abut,
    find_merger_candidates,
    extract_roi_to_ophys_roi,
    ophys_roi_to_extract_roi,
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
    for ii in range(30):
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


def test_extract_roi_to_ophys_roi():
    rng = np.random.RandomState(345)
    mask = rng.randint(0, 2, (9, 7)).astype(bool)
    roi = {'x': 5,
           'y': 6,
           'width': 7,
           'height': 9,
           'id': 991,
           'valid': True,
           'mask': [list(i) for i in mask]}

    ophys_roi = extract_roi_to_ophys_roi(roi)
    assert ophys_roi.x0 == roi['x']
    assert ophys_roi.y0 == roi['y']
    assert ophys_roi.height == roi['height']
    assert ophys_roi.width == roi['width']
    assert ophys_roi.roi_id == roi['id']
    assert ophys_roi.valid_roi and roi['valid']
    np.testing.assert_array_equal(ophys_roi.mask_matrix, mask)


def test_ophys_roi_to_extract_roi(example_roi_list):
    for roi_in in example_roi_list:
        roi_out = ophys_roi_to_extract_roi(roi_in)
        assert roi_out['x'] == roi_in.x0
        assert roi_out['y'] == roi_in.y0
        assert roi_out['width'] == roi_in.width
        assert roi_out['height'] == roi_in.height
        assert roi_out['id'] == roi_in.roi_id
        np.testing.assert_array_equal(roi_in.mask_matrix,
                                      roi_out['mask'])


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

    y0 = 19
    x0 = 16
    height = 6
    width = 4
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

    assert do_rois_abut(roi0, roi1, dpix=1.0)

    # just touching
    roi1 = OphysROI(x0=26,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert do_rois_abut(roi0, roi1, dpix=1.0)

    roi1 = OphysROI(x0=27,
                    y0=48,
                    height=height,
                    width=width,
                    mask_matrix=mask,
                    roi_id=1,
                    valid_roi=True)

    assert not do_rois_abut(roi0, roi1, dpix=1.0)

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
                                 img_data,
                                 whole_dataset['video'],
                                 3,
                                 2.0)

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
