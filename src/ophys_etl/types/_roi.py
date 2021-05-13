import sys
from typing import List

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


class DenseROI(TypedDict):
    id: int
    x: int
    y: int
    width: int
    height: int
    valid_roi: bool
    mask_matrix: List[List[bool]]
    max_correction_up: float
    max_correction_down: float
    max_correction_left: float
    max_correction_right: float
    mask_image_plane: int
    exclusion_labels: List[str]


class ExtractROI(TypedDict):
    """AllenSDK extract_traces expects this format
    """
    id: int
    x: int
    y: int
    width: int
    height: int
    valid: bool
    mask: List[List[bool]]
