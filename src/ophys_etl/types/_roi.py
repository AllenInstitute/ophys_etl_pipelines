from typing import Dict, Union, Set, Tuple
import sys
from typing import List
import numpy as np
import copy

if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict

from ophys_etl.utils.array_utils import get_cutout_indices, get_cutout_padding


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
        self._contour_mask = None
        self._area = None
        self._global_pixel_set = None
        self._global_pixel_array = None

        height_match = (self._mask_matrix.shape[0] == self._height)
        width_match = (self._mask_matrix.shape[1] == self._width)
        if not height_match or not width_match:
            msg = 'in OphysROI\n'
            msg += f'mask_matrix.shape: {self._mask_matrix.shape}\n'
            msg += f'height: {self._height}\nwidth: {self._width}\n'
            raise RuntimeError(msg)

        # calculate centroid
        cr = 0
        cc = 0
        n = 0
        for irow in range(self.height):
            for icol in range(self.width):
                if not self._mask_matrix[irow, icol]:
                    continue
                n += 1
                cr += irow
                cc += icol
        self._centroid_row = self._y0 + cr / n
        self._centroid_col = self._x0 + cc / n

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

    def _create_global_pixel_set(self):
        """
        Create the set of (row, col) tuples in
        global coordinates that make up this ROI
        """
        valid = np.argwhere(self._mask_matrix)
        self._global_pixel_set = set([(r + self._y0, c + self._x0)
                                      for r, c in valid])

        self._global_pixel_array = np.array([[r + self._y0, c + self._x0]
                                             for r, c in valid])

    @property
    def global_pixel_set(self) -> Set[Tuple[int, int]]:
        """
        Set of pixels in global (row, col) coordinates
        that are set to True for this ROI
        """
        if self._global_pixel_set is None:
            self._create_global_pixel_set()
        return self._global_pixel_set

    @property
    def global_pixel_array(self) -> np.ndarray:
        """
        np.ndarray of pixels in global (row, col) coordinates
        that are set to True for this ROI
        """
        if self._global_pixel_array is None:
            self._create_global_pixel_set()
        return self._global_pixel_array

    @property
    def area(self) -> int:
        if self._area is None:
            self._area = self._mask_matrix.sum()
        return self._area

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
    def centroid_y(self) -> float:
        return self._centroid_row

    @property
    def centroid_x(self) -> float:
        return self._centroid_col

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def bounding_box_center_y(self) -> int:
        return int(
            np.round(self._y0 + self._height / 2))

    @property
    def bounding_box_center_x(self) -> int:
        return int(
            np.round(self._x0 + self._width / 2))

    @property
    def valid_roi(self) -> bool:
        return self._valid_roi

    @valid_roi.setter
    def valid_roi(self, value):
        self._valid_roi = value

    @property
    def mask_matrix(self) -> np.ndarray:
        return copy.deepcopy(self._mask_matrix)

    def _construct_contour_mask(self):
        """
        Construct a mask of contour pixels
        """
        self._contour_mask = np.zeros(self._mask_matrix.shape,
                                      dtype=bool)
        nr = self._contour_mask.shape[0]
        nc = self._contour_mask.shape[1]
        for irow in range(nr):
            ir0 = irow - 1
            ir1 = irow + 1
            for icol in range(nc):
                if not self._mask_matrix[irow, icol]:
                    continue
                ic0 = icol - 1
                ic1 = icol + 1
                left = False
                right = False
                if ic0 >= 0 and self._mask_matrix[irow, ic0]:
                    left = True
                if ic1 < nc and self._mask_matrix[irow, ic1]:
                    right = True
                if not (left and right):
                    self._contour_mask[irow, icol] = True
                    continue

                above = False
                below = False
                if ir0 >= 0 and self._mask_matrix[ir0, icol]:
                    below = True
                if ir1 < nr and self._mask_matrix[ir1, icol]:
                    above = True
                if not (above and below):
                    self._contour_mask[irow, icol] = True

    @property
    def contour_mask(self) -> np.ndarray:
        if self._contour_mask is None:
            self._construct_contour_mask()
        return np.copy(self._contour_mask)

    def get_bounding_box_cutout(self, image: np.ndarray) -> np.ndarray:
        """Return a cutout from an image that matches the roi bounding box.
        """
        return image[self._y0:self._y0 + self._height,
                     self._x0:self._x0 + self._width]

    def get_centered_cutout(self,
                            image: np.ndarray,
                            height: int,
                            width: int) -> np.ndarray:
        """Get a cutout of arbitrary size centered on the bounding box
        centroid.

        Pad cutout with zeros if requested cutout extends outside of image.

        Parameters
        ----------
        image : numpy.ndarray, (N, M)
            Image to create cutout/thumbnail from.
        height : int
            Height(y) of output cutout image.
        width : int
            Width(x) of output cutout image.

        Returns
        -------
        cutout : numpy.ndarray
            Cutout of requested size centered on the bounding box center.
        """
        # Find the indices of the desired cutout in the image.
        row_indices = get_cutout_indices(self.bounding_box_center_y,
                                         image.shape[0],
                                         height)
        col_indices = get_cutout_indices(self.bounding_box_center_x,
                                         image.shape[1],
                                         width)
        # Get initial cutout.
        thumbnail = image[row_indices[0]:row_indices[1],
                          col_indices[0]:col_indices[1]]
        # Find if we need to pad the image.
        row_pad = get_cutout_padding(self.bounding_box_center_y,
                                     image.shape[0],
                                     height)
        col_pad = get_cutout_padding(self.bounding_box_center_x,
                                     image.shape[1],
                                     width)
        # Pad the cutout if needed.
        padding = (row_pad, col_pad)
        return np.pad(thumbnail,
                      pad_width=padding, mode="constant",
                      constant_values=0)
