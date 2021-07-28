from typing import List, Dict, Union, Tuple, Set
import itertools
import h5py
import copy
import numpy as np
import logging

from ophys_etl.utils.roi_masks import RoiMask
from ophys_etl.modules.trace_extraction.utils import \
        calculate_roi_and_neuropil_traces
import ophys_etl.modules.decrosstalk.decrosstalk_types as dc_types


logger = logging.getLogger(__name__)


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
        self._boundary_mask = None
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
        self._centroid_row = self._y0 + cr/n
        self._centroid_col = self._x0 + cc/n

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
        self._global_pixel_set = set([(r+self._y0, c+self._x0)
                                      for r, c in valid])

        self._global_pixel_array = np.array([[r+self._y0, c+self._x0]
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
    def valid_roi(self) -> bool:
        return self._valid_roi

    @valid_roi.setter
    def valid_roi(self, value):
        self._valid_roi = value

    @property
    def mask_matrix(self) -> np.ndarray:
        return copy.deepcopy(self._mask_matrix)

    def _construct_boundary_mask(self):
        """
        Construct a mask of boundary pixels
        """
        self._boundary_mask = np.zeros(self._mask_matrix.shape,
                                       dtype=bool)
        nr = self._boundary_mask.shape[0]
        nc = self._boundary_mask.shape[1]
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
                    self._boundary_mask[irow, icol] = True
                    continue

                above = False
                below = False
                if ir0 >= 0 and self._mask_matrix[ir0, icol]:
                    below = True
                if ir1 < nr and self._mask_matrix[ir1, icol]:
                    above = True
                if not (above and below):
                    self._boundary_mask[irow, icol] = True

    @property
    def boundary_mask(self) -> np.ndarray:
        if self._boundary_mask is None:
            self._construct_boundary_mask()
        return np.copy(self._boundary_mask)


class OphysMovie(object):

    def __init__(self, movie_path: str, motion_border: Dict[str, float]):
        """
        Parameters
        ----------
        movie_path -- path to the motion corrected movie file

        motion_border -- dict defining the border of the valid region
        within each frame, e.g.
            {
                "y1": 19.0758,
                "y0": 22.3232,
                "x0": 9.16988,
                "x1": 7.79272
             }
        """

        self._path = movie_path
        self._motion_border = copy.deepcopy(motion_border)

        # this is where the data from the movie file will be stored
        self._data = None
        self._max_rgb = None

    @property
    def path(self) -> str:
        return self._path

    @property
    def motion_border(self) -> Dict[str, float]:
        return copy.deepcopy(self._motion_border)

    def load_movie_data(self):
        """
        Load the data from self._path; store te data in self._data
        """
        logger.info(f'loading {self.path}')
        with h5py.File(self.path, mode='r') as in_file:
            self._data = in_file['data'][()]

    @property
    def data(self) -> np.ndarray:
        if self._data is None:
            self.load_movie_data()
        return self._data

    def purge_movie(self) -> None:
        """
        Delete loaded movie data
        """
        if self._data is not None:
            self._data = None
        return None

    def get_trace(self, roi_list: List[OphysROI]) -> dc_types.ROISetDict:
        """
        Extract the traces from a movie as defined by the ROIs in roi_list

        Parameters
        ----------
        roi_list -- a list of OphysROI instantiations
                    specifying the ROIs from which to
                    extract traces

        Returns
        -------
        output -- a decrosstalk_types.ROISetDict containing the ROI and
                  neuropil traces associated with roi_list. For each ROI
                  in the ROISetDict, only the 'signal' channel will be
                  populated, this with the trace extracted from the movie.
        """
        motion_border = [self._motion_border['x0'], self._motion_border['x1'],
                         self._motion_border['y0'], self._motion_border['y1']]

        height = self.data.shape[1]
        width = self.data.shape[2]

        roi_mask_list = []
        for roi in roi_list:
            pixels = np.argwhere(roi.mask_matrix)
            pixels[:, 0] += roi.y0
            pixels[:, 1] += roi.x0
            mask = RoiMask.create_roi_mask(width, height, motion_border,
                                           pix_list=pixels[:, [1, 0]],
                                           label=str(roi.roi_id),
                                           mask_group=-1)

            roi_mask_list.append(mask)

        _traces = calculate_roi_and_neuropil_traces(self.data,
                                                    roi_mask_list,
                                                    motion_border)
        roi_traces = _traces[0]
        neuropil_traces = _traces[1]

        output = dc_types.ROISetDict()
        for i_roi, roi in enumerate(roi_list):

            trace = dc_types.ROIChannels()
            trace['signal'] = roi_traces[i_roi]
            output['roi'][roi.roi_id] = trace

            trace = dc_types.ROIChannels()
            trace['signal'] = neuropil_traces[i_roi]
            output['neuropil'][roi.roi_id] = trace

        return output

    def _load_max_rgb(self,
                      keep_data: bool = False) -> np.ndarray:
        """
        Load the max RGB image of this movie

        Parameters
        ----------
        keep_data: bool
            If True, keep the movie data loaded after running this
            image. If False, purge movie data after running. If movie
            data was already loaded before running, data will not be
            purged, regardless. (Default: False)

        Returns
        -------
        max_img: np.ndarray
            A (nrows, ncols, 3) np.ndarray of ints representing
            the RGB form of the max projection image that goes with
            this movie
        """

        if self._data is not None:
            keep_data = True

        raw_max = self.data.max(axis=0)
        max_rgb = np.zeros((raw_max.shape[0],
                            raw_max.shape[1],
                            3),
                           dtype=int)

        max_val = raw_max.max()
        gray = np.round(255*(raw_max/max_val)).astype(int)
        gray = np.where(gray <= 255, gray, 255)
        for ic in range(3):
            max_rgb[:, :, ic] = gray

        if not keep_data:
            self.purge_movie()

        self._max_rgb = max_rgb

    def get_max_rgb(self,
                    keep_data: bool = False):
        """
        Get the maximum projection image of this movie as an RGB
        array

        Parameters
        ----------
        keep_data: bool
            If True, keep the movie data loaded after running this
            image. If False, purge movie data after running. If movie
            data was already loaded before running, data will not be
            purged, regardless. (Default: False)

        Returns
        -------
        max_img: np.ndarray
            A (nrows, ncols, 3) np.ndarray of ints representing
            the RGB form of the max projection image that goes with
            this movie
        """
        if self._max_rgb is None:
            self._load_max_rgb(keep_data=keep_data)
        return np.copy(self._max_rgb)


class DecrosstalkingOphysPlane(object):

    def __init__(self,
                 experiment_id=None,
                 movie_path=None,
                 motion_border=None,
                 roi_list=None,
                 max_projection_path=None,
                 qc_file_path=None):

        """
        Parameters
        ----------
        experiment_id -- an integer uniquely identifying
                         this experimental plane

        movie_path -- path to the motion corrected movie file

        motion_border -- dict defining the border of the valid region
        within each frame, e.g.
            {
                "y1": 19.0758,
                "y0": 22.3232,
                "x0": 9.16988,
                "x1": 7.79272
             }

        roi_list -- a list of OphysROIs indicating the ROIs in this movie

        max_projection_path -- path to the maximum projection image for
                               this plane

        qc_file_path -- path to the HDF5 file containing quality control
                        data. This will probably always be initialized
                        to None and will be set by the decrosstalking
                        pipeline after processing but before quality
                        control figure generation.
        """

        if experiment_id is None or not isinstance(experiment_id, int):
            raise ValueError("DecrosstalkingOphysPlane.experiment_id "
                             "must be an int; you gave "
                             "%s" % str(type(experiment_id)))

        if movie_path is None:
            raise ValueError("Must specify movie_path when "
                             "initializing DecrosstalkingOphysPlane")

        if motion_border is None or not isinstance(motion_border, dict):
            raise ValueError("DecrosstalkingOphysPlane.motion_border "
                             "must be a dict; you gave "
                             "%s" % str(type(motion_border)))

        if roi_list is None or not isinstance(roi_list, list):
            raise ValueError("DecrosstalkingOphysPlane.roi_list "
                             "must be a list of OphysROI; you gave "
                             "%s" % str(type(roi_list)))

        if len(roi_list) > 0:
            for roi in roi_list:
                if not isinstance(roi, OphysROI):
                    raise ValueError("DecrosstalkingOphysPlane.roi_list "
                                     "must be a list of OphysROI; you gave "
                                     "a list of %s" % str(type(roi)))

        self._experiment_id = experiment_id
        self._movie = OphysMovie(movie_path, motion_border)
        self._roi_list = copy.deepcopy(roi_list)
        self._max_projection_path = max_projection_path
        self._qc_file_path = qc_file_path

    @property
    def experiment_id(self) -> int:
        return self._experiment_id

    @property
    def movie(self) -> OphysMovie:
        return self._movie

    @property
    def roi_list(self) -> List[OphysROI]:
        return self._roi_list

    @property
    def maximum_projection_image_path(self) -> Union[None, str]:
        return self._max_projection_path

    @property
    def qc_file_path(self) -> Union[None, str]:
        return self._qc_file_path

    @qc_file_path.setter
    def qc_file_path(self, input_path: str):
        self._qc_file_path = input_path

    @classmethod
    def from_schema_dict(cls, schema_dict):
        """
        Create an OphysPlane from a dict taken from the module's argschema

        Parameters
        ----------
        schema_dict -- a dict codifying the plane, as read from argschema, i.e.

        {  # start of ophys_experiment
         "ophys_experiment_id": ,# an int
         "motion_corrected_stack": ,  # path to h5 movie file
         "motion_border": {  # border widths
                 "x0": ,  # a float
                 "x1": ,  # a float
                 "y0": ,  # a float
                 "y1": ,  # a float
           },
           "rois": [  # list of dicts definining the ROIs for this experiment
               {  # start of individual ROI
                 "id":  ,  # an int
                 "x": ,  # an int
                 "y": ,  # an int
                 "width": ,  # an int
                 "height": ,  # an int
                 "valid_roi": ,  # boolean
                 "mask_matrix": [[]]  # 2-D array of booleans
               },  # end of individual ROI,
               {
                 "id":  ,
                 "x": ,
                 "y": ,
                 "width": ,
                 "height": ,
                 "valid_roi": ,
                 "mask_matrix": [[]]
               },
               ...
           ]
         }
        """
        roi_list = []
        for roi in schema_dict['rois']:
            roi_list.append(OphysROI.from_schema_dict(roi))

        max_path = None
        if 'maximum_projection_image_file' in schema_dict:
            max_path = schema_dict['maximum_projection_image_file']

        return cls(experiment_id=schema_dict['ophys_experiment_id'],
                   movie_path=schema_dict['motion_corrected_stack'],
                   motion_border=schema_dict['motion_border'],
                   roi_list=roi_list,
                   max_projection_path=max_path)


def get_roi_pixels(roi_list: List[OphysROI]) -> Dict[int, set]:
    """
    Take a list of OphysROIs and return a dict
    that maps roi_id to a set of (x,y) pixel coordinates
    corresponding to the masks of the ROIs

    Parameters
    ----------
    roi_list: List[OphysROI]

    Returns
    -------
    roi_pixel_dict: dict
        A dict whose keys are the ROI IDs of the ROIs in the input
        plane and whose values are sets of tuples. Each tuple is
        an (x, y) pair denoting a pixel in the ROI's mask
    """

    roi_pixel_dict = {}
    for roi in roi_list:
        roi_id = roi.roi_id
        grid = np.meshgrid(roi.x0+np.arange(roi.width, dtype=int),
                           roi.y0+np.arange(roi.height, dtype=int))
        mask_arr = roi.mask_matrix.flatten()
        x_coords = grid[0].flatten()[mask_arr]
        y_coords = grid[1].flatten()[mask_arr]
        roi_pixel_dict[roi_id] = set([(x, y)
                                      for x, y
                                      in zip(x_coords, y_coords)])
    return roi_pixel_dict


def find_overlapping_roi_pairs(roi_list_0: List[OphysROI],
                               roi_list_1: List[OphysROI]
                               ) -> List[Tuple[int, int, float, float]]:
    """
    Find all overlapping pairs from two lists of OphysROIs

    Parameters
    ----------
    roi_list_0: List[OphysROI]

    roi_list_1: List[OphysROI]

    Return:
    -------
    overlapping_pairs: list
        A list of tuples. Each tuple contains
        roi_id_0
        roi_id_0
        fraction of roi_id_0 that overlaps roi_id_1
        fraction of roi_id_1 that overlaps roi_id_0
    """

    pixel_dict_0 = get_roi_pixels(roi_list_0)
    pixel_dict_1 = get_roi_pixels(roi_list_1)

    overlapping_pairs = []

    roi_id_list_0 = list(pixel_dict_0.keys())
    roi_id_list_1 = list(pixel_dict_1.keys())

    for roi_pair in itertools.product(roi_id_list_0,
                                      roi_id_list_1):
        roi0 = pixel_dict_0[roi_pair[0]]
        roi1 = pixel_dict_1[roi_pair[1]]
        overlap = roi0.intersection(roi1)
        n = len(overlap)
        if n > 0:
            datum = (roi_pair[0], roi_pair[1], n/len(roi0), n/len(roi1))
            overlapping_pairs.append(datum)
    return overlapping_pairs
