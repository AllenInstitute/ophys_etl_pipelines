import pytest
import h5py
import numpy as np
import copy

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.modules.filter_area import (
    AreaFilterRunner)
from ophys_etl.modules.segmentation.utils.roi_utils import \
    ophys_roi_to_extract_roi
from ophys_etl.modules.segmentation.processing_log import \
    SegmentationProcessingLog


def compare_rois(roi0: OphysROI,
                 roi1: OphysROI) -> bool:
    """
    Compare two OphysROIs (ignoring valid_roi).
    Return True if they are identical. Return False
    otherwise.

    Parameters
    ----------
    roi0: OphysROI

    roi1: OphysROI

    Returns
    -------
    bool
    """
    if roi0.x0 != roi1.x0:
        return False
    if roi0.y0 != roi1.y0:
        return False
    if roi0.width != roi1.width:
        return False
    if roi0.height != roi1.height:
        return False
    if roi0.roi_id != roi1.roi_id:
        return False
    if not np.array_equal(roi0.mask_matrix, roi1.mask_matrix):
        return False

    return True


@pytest.fixture
def log_file_with_previous_step(tmpdir, roi_dict, seeder_fixture):
    path = tmpdir / "log_with_step.h5"
    rois = [ophys_roi_to_extract_roi(i) for i in roi_dict.values()]
    log = SegmentationProcessingLog(path, read_only=False)
    log.log_detection(attribute="something",
                      rois=rois,
                      group_name="previous_step",
                      seeder=seeder_fixture,
                      seeder_group_name="seed")
    yield str(path)


@pytest.mark.parametrize(
        'min_area, max_area, valid_roi',
        [(4, 10, {2, 3, 4, 5}),
         (None, 8, set([1, 2, 3, 4])),
         (6, None, set([3, 4, 5, 6]))
         ])
def test_area_filter_runner(log_file_with_previous_step,
                            min_area,
                            max_area,
                            valid_roi):
    input_args = {'log_path': log_file_with_previous_step,
                  'min_area': min_area,
                  'max_area': max_area,
                  'pipeline_stage': 'unit test'}

    runner = AreaFilterRunner(input_data=input_args,
                              args=[])
    runner.run()

    log = SegmentationProcessingLog(log_file_with_previous_step,
                                    read_only=True)
    original_rois = log.get_rois_from_group("previous_step")
    filtered_rois = log.get_rois_from_group("filter")
    valid_filtered = {i['id'] for i in filtered_rois if i["valid"]}
    # filter should not delete ROIs
    assert len(original_rois) == len(filtered_rois)
    # expected valid ROIs
    assert valid_filtered == valid_roi

    original_lookup = {i['id']: i for i in original_rois}
    filtered_lookup = {i['id']: i for i in original_rois}
    for k, v in filtered_lookup.items():
        original = copy.deepcopy(original_lookup[k])
        if not v["valid"]:
            # if invalid, pop this key for comparison
            was_valid = original.pop("valid_roi")
            is_valid = v.pop("valid_roi")
            assert was_valid & (not is_valid)
        # except for validity, filtering should not modify contents:
        assert original == v

    invalid_filtered = set(original_lookup.keys()) - valid_filtered
    with h5py.File(log.path, "r") as f:
        print(f.keys())
        group = f["filter"]
        # does the log file show the right filtered IDs
        assert set(group["filter_ids"][()]) == invalid_filtered
        # did the name get logged
        assert (group["filter_reason"][()].decode("utf-8") ==
                "area -- unit test")
