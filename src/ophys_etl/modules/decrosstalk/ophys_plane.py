from typing import List, Dict, Union
import h5py
import copy
import numpy as np

from ophys_etl.utils.roi_masks import RoiMask
from ophys_etl.modules.trace_extraction.utils import \
        calculate_roi_and_neuropil_traces
import ophys_etl.modules.decrosstalk.decrosstalk_types as dc_types


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
        if (self._mask_matrix.shape[0] != self._height or
            self._mask_matrix.shape[1] != self._width):  # noqa E129

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
        print(f'loading {self.path}')
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
            del self._data
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
