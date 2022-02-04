from typing import Dict, Union
import sys
from typing import List
import numpy as np
import copy

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


class OphysROI(object):

    def __init__(self,
                 roi_id=None,
                 x0=None,
                 y0=None,
                 width=None,
                 height=None,
                 valid_roi=None,
                 mask_matrix=None):
        """
        Parameters
        ----------
        roi_id -- an integer identifying the ROI. Unique within the context
        of a specific experiment_id

        x0 -- an integer defining the starting x pixel of the mask_array

        y0 -- an integer defining the starting y pixel of the mask_array

        width -- an integer defining the width of the mask_array

        height -- an integer defining the width of the mask_array

        valid_roi -- a boolean indicating the validity of the ROI

        mask_matrix -- a list of lists of booleans defining the pixels
        that are a part of the ROI
        """

        if roi_id is None or not isinstance(roi_id, int):
            raise ValueError("OphysROI.roi_id must be an int; "
                             "you gave %s" % str(type(roi_id)))

        if x0 is None or not isinstance(x0, int):
            raise ValueError("OphysROI.x0 must be an int; "
                             "you gave %s" % str(type(x0)))

        if y0 is None or not isinstance(y0, int):
            raise ValueError("OphysROI.y0 must be an int; "
                             "you gave %s" % str(type(y0)))

        if width is None or not isinstance(width, int):
            raise ValueError("OphysROI.width must be an int; "
                             "you gave %s" % str(type(width)))

        if height is None or not isinstance(height, int):
            raise ValueError("OphysROI.x0 must be an int; "
                             "you gave %s" % str(type(height)))

        if valid_roi is None or not isinstance(valid_roi, bool):
            raise ValueError("OphysROI.valid_roi must be a bool; "
                             "you gave %s" % str(type(valid_roi)))

        if (mask_matrix is None
            or (not isinstance(mask_matrix, list)
                and not isinstance(mask_matrix, np.ndarray))):

            raise ValueError("OphysROI.mask_matrix must be a list or array; "
                             "you gave %s" % str(type(mask_matrix)))

        self._roi_id = roi_id
        self._x0 = x0
        self._y0 = y0
        self._width = width
        self._height = height
        self._valid_roi = valid_roi
        self._mask_matrix = np.array(mask_matrix, dtype=bool)

        height_match = (self._mask_matrix.shape[0] == self._height)
        width_match = (self._mask_matrix.shape[1] == self._width)
        if not height_match or not width_match:
            msg = 'in OphysROI\n'
            msg += f'mask_matrix.shape: {self._mask_matrix.shape}\n'
            msg += f'height: {self._height}\nwidth: {self._width}\n'
            raise RuntimeError(msg)

    @classmethod
    def from_schema_dict(cls, schema_dict: Dict[str, Union[int, List]]):
        """
        Create an OphysROI from the argschema dict associated with the
        decrosstalk pipeline, i.e.

        {  # start of individual ROI
           "id":  ,  # an int
           "x": ,  # an int
           "y": ,  # an int
           "width": ,  # an int
           "height": ,  # an int
           "valid_roi": ,  # boolean
           "mask_matrix": [[]]  # 2-D array of booleans
        }
        """

        return cls(roi_id=schema_dict['id'],
                   x0=schema_dict['x'],
                   y0=schema_dict['y'],
                   width=schema_dict['width'],
                   height=schema_dict['height'],
                   valid_roi=schema_dict['valid_roi'],
                   mask_matrix=schema_dict['mask_matrix'])

    @property
    def roi_id(self) -> int:
        return self._roi_id

    @property
    def x0(self) -> int:
        return self._x0

    @property
    def y0(self) -> int:
        return self._y0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def valid_roi(self) -> bool:
        return self._valid_roi

    @property
    def mask_matrix(self) -> np.ndarray:
        return copy.deepcopy(self._mask_matrix)
