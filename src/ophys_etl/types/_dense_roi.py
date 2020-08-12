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
    max_correction_up: int
    max_correction_down: int
    max_correction_left: int
    max_correction_right: int
    mask_image_plane: int
    exclusion_labels: List[str]
