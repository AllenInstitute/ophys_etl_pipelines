import pytest
import numpy as np
from ophys_etl.modules.segmentation.merge.candidates import (
    find_merger_candidates,
    create_neighbor_lookup)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    do_rois_abut)


@pytest.mark.parametrize("pixel_distance", [np.sqrt(2), 4, 5])
def test_find_merger_candidates(pixel_distance, example_roi_list):
    true_matches = set()
    has_been_matched = set()
    for i0 in range(len(example_roi_list)):
        roi0 = example_roi_list[i0]
        for i1 in range(i0+1, len(example_roi_list), 1):
            roi1 = example_roi_list[i1]
            if do_rois_abut(roi0, roi1, pixel_distance=pixel_distance):
                true_matches.add((roi0.roi_id, roi1.roi_id))
                has_been_matched.add(roi0.roi_id)
                has_been_matched.add(roi1.roi_id)

    assert len(has_been_matched) > 0
    assert len(has_been_matched) < len(example_roi_list)
    expected = set(true_matches)

    for n in (3, 5):
        matches = find_merger_candidates(example_roi_list,
                                         pixel_distance,
                                         n_processors=5)
        matches = set(matches)
        assert matches == expected


@pytest.mark.parametrize("pixel_distance", [np.sqrt(2), 4, 5])
def test_find_merger_candidates_with_ignore(pixel_distance, example_roi_list):
    full_matches = set()
    has_been_matched = set()
    for i0 in range(len(example_roi_list)):
        roi0 = example_roi_list[i0]
        for i1 in range(i0+1, len(example_roi_list), 1):
            roi1 = example_roi_list[i1]
            if do_rois_abut(roi0, roi1, pixel_distance=pixel_distance):
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
                                         pixel_distance,
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


@pytest.mark.parametrize('n_processors', [2, 3, 5])
def test_create_neighbor_lookup(example_roi_list, n_processors):
    true_matches = find_merger_candidates(example_roi_list,
                                          np.sqrt(2),
                                          rois_to_ignore=None,
                                          n_processors=n_processors)

    neighbor_lookup = create_neighbor_lookup(
                        {roi.roi_id: roi for roi in example_roi_list},
                        n_processors)

    for match in true_matches:
        assert match[0] in neighbor_lookup[match[1]]
        assert match[1] in neighbor_lookup[match[0]]

    for id0 in neighbor_lookup:
        for id1 in neighbor_lookup[id0]:
            assert (id0, id1) in true_matches or (id1, id0) in true_matches
