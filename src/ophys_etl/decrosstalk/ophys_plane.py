import h5py
import copy
import numpy as np
import ophys_etl.decrosstalk.roi_masks as roi_masks


class OphysROI(object):

    def __init__(self, roi_id=None, x0=None, y0=None,
                 width=None, height=None, valid_roi=None,
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

        self._roi_id = roi_id
        self._x0 = x0
        self._y0 = y0
        self._width = width
        self._height = height
        self._valid_roi = valid_roi
        self._mask_matrix = np.array(mask_matrix, dtype=bool)

    @classmethod
    def from_schema_dict(cls, schema_dict):
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
    def roi_id(self):
        return self._roi_id

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def valid_roi(self):
        return self._valid_roi

    @property
    def mask_matrix(self):
        return copy.deepcopy(self._mask_matrix)


class OphysMovie(object):

    def __init__(self, movie_path, motion_border):
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
    def path(self):
        return self._path

    @property
    def motion_border(self):
        return copy.deepcopy(self._motion_border)

    def load_movie_data(self):
        """
        Load the data from self._path; store te data in self._data
        """
        with h5py.File(self.path, mode='r') as in_file:
            self._data = in_file['data'][()]

    @property
    def data(self):
        if self._data is None:
            self.load_movie_data()
        return self._data

    def get_trace(self, roi_list):
        """
        Extract the traces from a movie as defined by the ROIs in roi_list

        Parameters
        ----------
        roi_list -- a list of OphysROI instantiations
                    specifying the ROIs from which to
                    extract traces

        Returns
        -------
        output -- a dict such that

            output['roi'][roi_id] = np.array of trace values for the ROI

            output['neuropil'][roi_id] = np.array of trace values defined
                                         in the neuropil around the ROI
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
            mask = roi_masks.create_roi_mask(width, height, motion_border,
                                             pix_list=pixels[:, [1, 0]],
                                             label=str(roi.roi_id),
                                             mask_group=-1)

            roi_mask_list.append(mask)

        _traces = roi_masks.calculate_roi_and_neuropil_traces(self.data,
                                                              roi_mask_list,
                                                              motion_border)
        roi_traces = _traces[0]
        neuropil_traces = _traces[1]

        output = {}
        output['roi'] = {}
        output['neuropil'] = {}
        for i_roi, roi in enumerate(roi_list):
            output['roi'][roi.roi_id] = roi_traces[i_roi]
            output['neuropil'][roi.roi_id] = neuropil_traces[i_roi]
        return output


class OphysPlane(object):

    def __init__(self,
                 experiment_id=None,
                 movie_path=None,
                 motion_border=None,
                 roi_list=None):

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
        """

        self._experiment_id = experiment_id
        self._movie = OphysMovie(movie_path, motion_border)
        self._roi_list = copy.deepcopy(roi_list)

    @property
    def experiment_id(self):
        return self._experiment_id

    @property
    def movie(self):
        return self._movie

    @property
    def roi_list(self):
        return self._roi_list

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

        return cls(experiment_id=schema_dict['ophys_experiment_id'],
                   movie_path=schema_dict['motion_corrected_stack'],
                   motion_border=schema_dict['motion_border'],
                   roi_list=roi_list)
