import pytest
import numpy as np
import copy
from itertools import combinations
from ophys_etl.modules.segmentation.merge.roi_merging import (
    do_roi_merger,
    get_new_merger_candidates,
    break_out_anomalous_rois,
    update_lookup_tables)


@pytest.fixture
def lookup_tables(example_roi_list):
    rng = np.random.RandomState(99)
    roi_lookup = {roi.roi_id: roi for roi in example_roi_list}
    timeseries_lookup = {roi.roi_id: {'area': roi.area,
                                      'timeseries': rng.random_sample(22)}
                         for roi in example_roi_list}

    neighbor_lookup = dict()
    for i0 in range(0, 30, 10):
        i1 = i0+10
        for ii in range(i0, i1, 1):
            neighbor_lookup[ii] = set([jj for jj in range(i0, i1, 1)
                                       if jj!=ii])

    sub_video_lookup = {roi.roi_id: rng.random_sample((22,22))
                        for roi in example_roi_list}

    pairs = [(ii, ii+1) for ii in range(0, len(example_roi_list), 2)]
    merger_to_metric = {p: rng.random_sample() for p in pairs}

    return(roi_lookup,
           neighbor_lookup,
           timeseries_lookup,
           sub_video_lookup,
           merger_to_metric)


def test_update_lookup_tables(lookup_tables):
    roi_lookup = copy.deepcopy(lookup_tables[0])
    neighbor_lookup = copy.deepcopy(lookup_tables[1])
    timeseries_lookup = copy.deepcopy(lookup_tables[2])
    sub_video_lookup = copy.deepcopy(lookup_tables[3])
    merger_to_metric = copy.deepcopy(lookup_tables[4])

    # test case where an ROI is recently merged
    recently_merged = set([18])
    (new_neighbors,
     new_video,
     new_timeseries,
     new_merger) = update_lookup_tables(roi_lookup,
                                        recently_merged,
                                        neighbor_lookup,
                                        sub_video_lookup,
                                        timeseries_lookup,
                                        merger_to_metric)

    assert 18 in sub_video_lookup
    assert 18 not in new_video
    is_there = False
    for pair in merger_to_metric:
        if 18 in pair:
            is_there = True
            break
    assert is_there
    for pair in new_merger:
        assert 18 not in pair

    # test case where an ROI is missing from roi_lookup
    altered_roi_lookup = copy.deepcopy(roi_lookup)
    altered_roi_lookup.pop(22)
    (new_neighbors,
     new_video,
     new_timeseries,
     new_merger) = update_lookup_tables(altered_roi_lookup,
                                        set(),
                                        neighbor_lookup,
                                        sub_video_lookup,
                                        timeseries_lookup,
                                        merger_to_metric)

    assert 22 in neighbor_lookup
    assert 22 not in new_neighbors
    assert 22 in sub_video_lookup
    assert 22 not in new_video
    assert 22 in timeseries_lookup
    assert 22 not in new_timeseries
    is_there = False
    for pair in merger_to_metric:
        if 22 in pair:
            is_there = True
    assert is_there
    for pair in new_merger:
        assert 22 not in pair


@pytest.mark.parametrize('anomalous_size', [5, 7, 12])
def test_break_out_anomalous_rois(example_roi_list, anomalous_size):
    roi_lookup = {roi.roi_id: roi for roi in example_roi_list}
    anomalous_rois = dict()
    (roi_lookup,
     anomalous_rois) = break_out_anomalous_rois(roi_lookup,
                                                anomalous_rois,
                                                anomalous_size)

    assert len(roi_lookup) > 0
    assert len(anomalous_rois) > 0
    for roi in example_roi_list:
        if roi.area >= anomalous_size:
            assert roi.roi_id in anomalous_rois
        else:
            assert roi.roi_id in roi_lookup


def test_get_new_merger_candidates():

    merger_to_metric = {(11, 9): 4.2,
                        (13, 7): 3.1}

    neighbor_lookup = dict()
    neighbor_lookup[11] = [9, 4, 5, 15]
    neighbor_lookup[7] = [13, 4, 6]

    expected = {(11, 4), (11, 5), (15, 11),
                (7, 4), (7, 6)}

    k_list = list(neighbor_lookup.keys())
    for k in k_list:
        for n in neighbor_lookup[k]:
            if n not in neighbor_lookup:
                neighbor_lookup[n] = []
            neighbor_lookup[n].append(k)

    new_candidates = set(get_new_merger_candidates(neighbor_lookup,
                                                   merger_to_metric))
    assert new_candidates == expected


def test_do_roi_merger(roi_and_video_dataset):
    """
    smoke test for do_roi_merger
    """
    img_data = np.mean(roi_and_video_dataset['video'], axis=0)
    assert img_data.shape == roi_and_video_dataset['video'].shape[1:]
    new_roi_list = do_roi_merger(roi_and_video_dataset['roi_list'],
                                 roi_and_video_dataset['video'],
                                 3,
                                 2.0,
                                 filter_fraction=0.2)

    # test that some mergers were performed
    assert len(new_roi_list) > 0
    assert len(new_roi_list) < len(roi_and_video_dataset['roi_list'])

    # check that pixels were conserved
    input_pixels = set()
    for roi in roi_and_video_dataset['roi_list']:
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
