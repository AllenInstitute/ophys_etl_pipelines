from typing import List, Tuple, Union, Optional
from scipy.sparse import coo_matrix
import numpy as np
import cv2

import lims.query_utils as query_utils
from ophys_etl.transforms.utils.array_utils import center_pad_2d, crop_2d_array


def coo_from_lims_style(mask_matrix: List[List[bool]],
                        xoffset: int = 0,
                        yoffset: int = 0,
                        shape: Optional[Tuple] = None) -> coo_matrix:
    """creates a coo matrix from a LIMS-style specification

    Parameters
    ----------
    mask_matrix: list(list(bool))
        boolean array of included pixels in bounding box of ROI
    xoffset: int
        column index of first pixel in mask_matrix
    yoffset: int
        row index of first pixel in mask_matrix
    shape: tuple(int)
        desired full-frame shape of coo_matrix. If left as default
        None, then the coo_matrix will extend from (0, 0) to the
        extent of the offset bounding box

    Returns
    -------
    coo: coo_matrix
        a scipy.sparse.coo_matrix

    """
    bounded = coo_matrix(mask_matrix)
    bounded.row += yoffset
    bounded.col += xoffset
    # re-constructing allows easy passing of None to shape
    coo = coo_matrix(
            (bounded.data, (bounded.row, bounded.col)),
            shape=shape)
    return coo


def binary_mask_from_threshold(
            arr: Union[np.ndarray, coo_matrix],
            absolute_threshold: float = None,
            quantile: float = 0.1) -> np.array:
    """Binarize an array

    Parameters
    ----------
    arr: numpy.ndarray or scipy.sparse.coo_matrix
        2D array of weighted floating-point values
    absolute_threshold: float
        weighted entries above this value=1, below=0
        over-ridden if quantile is set
    quantile: float
        if specified, set absolute threshold np.quantile(arr.data, quantile)

    Returns
    -------
    binary: numpy.ndarray
        binarized mask

    """
    wmask = arr
    if isinstance(arr, coo_matrix):
        wmask = arr.toarray()
    vals = wmask[np.nonzero(wmask)]

    if quantile is not None:
        absolute_threshold = np.quantile(vals, quantile)

    binary = np.uint8(wmask > absolute_threshold)

    return binary


def sized_mask(
        arr: Union[np.ndarray, coo_matrix], shape: Tuple[int, int] = None,
        full: bool = False):
    """return a 2D dense array representation of the mask, optionally
    cropped and padded

    Parameters
    ----------
    arr: numpy.ndarray or scipy.sparse.coo_matrix:
        a representation of the mask
    shape: tuple(int, int)
        [h, w] for padded shape. If None, cropped to existing values
    full: bool
        if True, the full-frame array is returned

    Returns
    -------
    mask: numpy.ndarray
        2D dense matrix representation of the mask

    """
    if isinstance(arr, coo_matrix):
        mask = arr.toarray()
    else:
        mask = arr
    if not full:
        mask = crop_2d_array(mask)
        if shape is not None:
            mask = center_pad_2d(mask, shape)
    return mask


class ROI:
    """Class is used for manipulating ROI from LIMs for serving to labeling app

    This class is used for loading the ROIs from LIMs DB tables and
    contains pre processing methods for ROIs loaded from LIMs. These methods
    are used to define drawing parameters as well as other ROI class
    methods useful for post processing required to display to end user.
    Attributes:
        image_shape: the shape of the image the roi is contained within
        experiment_id: the unique id for the segmentation run
        roi_id: the unique id for the ROI in the segmentation run
        _sparse_coo: the sparse matrix containing the probability mask
        for the ROI
    """

    def __init__(self,
                 coo_rows: Union[np.array, List[int]],
                 coo_cols: Union[np.array, List[int]],
                 coo_data: Union[np.array, List[float]],
                 image_shape: Tuple[int, int],
                 experiment_id: int,
                 roi_id: int,
                 trace: Union[np.array, List[float], None],
                 is_binary: bool = False):
        self.image_shape = image_shape
        self.experiment_id = experiment_id
        self.roi_id = roi_id
        self._sparse_coo = coo_matrix((coo_data, (coo_rows, coo_cols)),
                                      shape=image_shape)
        self.trace = trace
        self.is_binary = is_binary

    @classmethod
    def roi_from_query(cls, roi_id: int,
                       db_conn: query_utils.DbConnection) -> "ROI":
        """
        Queries and builds ROI object by querying LIMS table for
        produced labeling ROIs.
        Args:
            roi_id: Unique Id of the ROI to be loaded

        Returns: ROI object for the given segmentation_id and roi_id
        """

        roi = db_conn.query(f"SELECT * FROM rois WHERE id={roi_id}")[0]

        segmentation_run = db_conn.query(
            ("SELECT * FROM segmentation_runs WHERE "
             f"id={roi['segmentation_run_id']}"))[0]

        return ROI(coo_rows=roi['coo_row'],
                   coo_cols=roi['coo_col'],
                   coo_data=roi['coo_data'],
                   image_shape=segmentation_run['video_shape'],
                   experiment_id=segmentation_run['ophys_experiment_id'],
                   roi_id=roi_id,
                   trace=roi['trace']
                   )

    def generate_ROI_mask(
            self, shape: Tuple[int, int] = None, full: bool = False):
        """return a 2D dense representation of the mask

        Parameters
        ----------
        shape: tuple(int, int)
            [h, w] for padded shape. If None, cropped to existing values
        full: bool
            if True, the full-frame array is returned

        Returns
        -------
        mask: numpy.ndarray
            2D dense representation of the mask

        """
        mask = sized_mask(self._sparse_coo.toarray(), shape=shape, full=full)

        return mask

    def generate_ROI_outline(
            self, shape: Tuple[int, int] = None,
            full: bool = False,
            absolute_threshold: float = None,
            quantile: float = 0.1,
            dilation_kernel_size: int = 1, inner_outline: bool = True):
        """return a 2D dense representation of the mask outline. Black (0)
        outline on a white (255) background.


        Parameters
        ----------
        shape: tuple(int, int)
            [h, w] for padded shape. If None, cropped to existing values
        full: bool
            if True, the full-frame array is returned
        absolute_threshold: float
            weighted entries above this value=1, below=0
            over-ridden if quantile is set
        quantile: float
            if specified, set absolute threshold by using np.quantile()
            with this value as the quantile arg
        dilation_kernel_size: int
            passed as size to cv2.getStructuringElement()
        inner_outline: bool
            whether to retain only outline points interior to the mask

        Returns
        -------
        mask: numpy.ndarray
            uint8 2D dense representation of the mask outline.

        """
        if self.is_binary:
            binary = self._sparse_coo.toarray()
        else:
            binary = binary_mask_from_threshold(
                self._sparse_coo,
                absolute_threshold=absolute_threshold,
                quantile=quantile)

        contours, _ = cv2.findContours(binary,
                                       cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)

        xy = np.concatenate(contours).squeeze()

        mask = np.full(binary.shape, False, dtype='uint8')
        mask[xy[:, 1], xy[:, 0]] = 1

        kernel = np.ones((dilation_kernel_size, dilation_kernel_size))

        mask = cv2.dilate(mask, kernel)

        if inner_outline:
            mask = mask & binary

        mask = sized_mask(mask, shape=shape, full=full)

        # convert to 0 outline on 255 background
        mask = 255 * (1 - mask)

        return mask