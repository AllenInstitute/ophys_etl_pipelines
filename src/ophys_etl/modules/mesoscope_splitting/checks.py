import sys
import numpy as np
from typing import List
from pathlib import Path

from ophys_etl.modules.mesoscope_splitting.tiff import tiff_header_data

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class MultiException(Exception):
    def __init__(self, exceptions):
        self.exceptions = exceptions

    def __str__(self):
        s = [f"{i.__class__.__name__} {i.args[0]}" for i in self.exceptions]
        return "\n".join([""] + s)


class ConsistencyInput(TypedDict):
    tiff: Path
    roi_index: List[int]


def splitting_consistency_check(check_list: List[ConsistencyInput]):
    """checks consistency between roi_index from the input json
    and the targeted roi index from the tiff header

    Parameters
    ----------
    check_list: List[ConsistencyInput]
        compiled in __main__ from the input json

    Notes
    -----
    the input json indicates for each tiff which roi_index to target
    in "plane_groups"/"ophys_experiments"/"roi_index"
    and the tiff header indicates the same where 'discretePlaneMode' = 0

    """
    errors = []
    for item_check in check_list:
        if len(item_check["roi_index"]) != 1:
            errors.append(
                    ValueError("expected each tiff to target a single "
                               f"'roi_index'. {item_check['tiff']} has these "
                               f"entries {item_check['roi_index']}"))
            continue
        header = tiff_header_data(item_check["tiff"])
        rois = header[1]["RoiGroups"]["imagingRoiGroup"]["rois"]
        modes = np.array([i["discretePlaneMode"] for i in rois])
        targeted_roi_indices = np.argwhere(modes == 0).flatten()
        if targeted_roi_indices.size != 1:
            errors.append(
                    ValueError("expected each tiff header data to indicate "
                               "one and only one 'discretePlaneMode' == 0. "
                               f"{item_check['tiff']} has these values that "
                               f"meet that criteria {targeted_roi_indices}"))
            continue
        if targeted_roi_indices[0] != item_check["roi_index"][0]:
            errors.append(
                    ValueError("expected the input json 'roi_index' "
                               f"( = {item_check['roi_index'][0]}) to match "
                               "the tiff header index with 'discretePlaneMode'"
                               f"== 0 ( = {targeted_roi_indices[0]}) for "
                               f"{item_check['tiff']}"))
    if len(errors) == 0:
        return
    if len(errors) == 1:
        raise errors[0]
    else:
        raise MultiException(errors)
