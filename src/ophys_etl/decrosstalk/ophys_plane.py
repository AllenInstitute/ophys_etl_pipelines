import h5py
import copy
import numpy as np

import logging

import ophys_etl.decrosstalk.decrosstalk_utils as d_utils
import ophys_etl.decrosstalk.ica_utils as ica_utils
import ophys_etl.decrosstalk.roi_masks as roi_masks
import ophys_etl.decrosstalk.io_utils as io_utils
import ophys_etl.decrosstalk.active_traces as active_traces

logger = logging.getLogger(__name__)


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

        self._trace_threshold_params = {'len_ne': 20, 'th_ag': 14}
        self._experiment_id = experiment_id
        self._movie = OphysMovie(movie_path, motion_border)
        self._roi_list = copy.deepcopy(roi_list)
        self.new_style_output = False

    @property
    def trace_threshold_params(self):
        return copy.deepcopy(self._trace_threshold_params)

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

    def run_decrosstalk(self, other_plane, cache_dir=None, clobber=False):
        """
        Actually run the decrosstalking pipeline against another OphysPlane

        Parameters
        ----------
        other_plane -- the OphysPlane characterizing the crosstalk for this
        plane

        cache_dir -- the directory in which to write the QC output
        (if None, the output does not get written)

        clobber -- a boolean indicating whether or not to overwrite
        pre-existing output files (default: False)

        Returns
        -------
        A dict listing the ROI IDs of ROIs that were ruled invalid
        for different reasons, namely:
            'decrosstalk_ghost_roi_ids' -- ROIs that are ghosts

            'decrosstalk_raw_exlucion_lable' -- ROIs with invalid
                                                raw traces

            'decrosstalk_unmixed_exclusion_label' -- ROIs with invalid
                                                     unmixed traces
        """

        # kwargs for output classes that write QC output
        output_kwargs = {'cache_dir': cache_dir,
                         'signal_plane': self,
                         'crosstalk_plane': other_plane,
                         'clobber': clobber}

        final_output = {}

        ghost_key = 'decrosstalk_ghost_roi_ids'
        raw_key = 'decrosstalk_raw_exclusion_label'
        unmixed_key = 'decrosstalk_unmixed_exclusion_label'

        final_output[ghost_key] = []
        final_output[raw_key] = []
        final_output[unmixed_key] = []

        ###############################
        # extract raw traces

        raw_traces = self.get_raw_traces(other_plane)

        if cache_dir is not None:
            if self.new_style_output:
                writer_class = io_utils.RawH5Writer
            else:
                writer_class = io_utils.RawH5WriterOld
            writer = writer_class(data=raw_traces, **output_kwargs)
            writer.run()
            del writer
            del writer_class

        raw_trace_validation = d_utils.validate_traces(raw_traces)
        if cache_dir is not None:
            if self.new_style_output:
                writer_class = io_utils.ValidJsonWriter
            else:
                writer_class = io_utils.ValidJsonWriterOld
            writer = writer_class(data=raw_trace_validation, **output_kwargs)
            writer.run()
            del writer
            del writer_class

        # remove invalid raw traces
        invalid_raw_trace = []
        for roi_id in raw_trace_validation:
            if not raw_trace_validation[roi_id]:
                invalid_raw_trace.append(roi_id)
                raw_traces['roi'].pop(roi_id)
                raw_traces['neuropil'].pop(roi_id)
        final_output[raw_key] += invalid_raw_trace

        if len(raw_traces['roi']) == 0:
            msg = 'No raw traces were valid when applying '
            msg += 'decrosstalk to ophys_experiment_id: '
            msg += '%d (%d)' % (self.experiment_id,
                                other_plane.experiment_id)
            logger.error(msg)
            return final_output

        #########################################
        # detect activity in raw traces

        raw_trace_events = self.get_trace_events(raw_traces['roi'])

        # For each ROI, calculate a random seed based on the flux
        # in all timestamps *not* chosen as events (presumably,
        # random noise)
        roi_to_seed = {}
        two_to_32 = 2**32
        for roi_id in raw_trace_events:
            flux_mask = np.ones(len(raw_traces['roi'][roi_id]['signal']),
                                dtype=bool)
            flux_mask[raw_trace_events[roi_id]['signal']['events']] = False
            _flux = np.abs(raw_traces['roi'][roi_id]['signal'][flux_mask])
            flux_sum = np.round(_flux.sum()).astype(int)
            roi_to_seed[roi_id] = flux_sum % two_to_32

        if cache_dir is not None:
            if self.new_style_output:
                writer_class = io_utils.RawATH5Writer
            else:
                writer_class = io_utils.RawATH5WriterOld
            writer = writer_class(data=raw_trace_events, **output_kwargs)
            writer.run()
            del writer
            del writer_class

        raw_trace_crosstalk_ratio = self.get_crosstalk_data(raw_traces['roi'],
                                                            raw_trace_events)

        del raw_trace_events

        ###########################################################
        # use Independent Component Analysis to separate out signal
        # and crosstalk

        (ica_converged,
         unmixed_traces) = self.unmix_all_ROIs(raw_traces,
                                               seed_lookup=roi_to_seed)
        if cache_dir is not None:
            if self.new_style_output:
                writer_class = io_utils.OutH5Writer
            else:
                writer_class = io_utils.OutH5WriterOld
            writer = writer_class(data=unmixed_traces, **output_kwargs)
            writer.run()
            del writer
            del writer_class

        if not ica_converged:
            msg = 'ICA did not converge for any ROIs when '
            msg += 'applying decrosstalk to ophys_experiment_id: '
            msg += '%d (%d)' % (self.experiment_id,
                                other_plane.experiment_id)
            logger.error(msg)
            return final_output

        unmixed_trace_validation = d_utils.validate_traces(unmixed_traces)
        if cache_dir is not None:
            if self.new_style_output:
                writer_class = io_utils.OutValidJsonWriter
            else:
                writer_class = io_utils.OutValidJsonWriterOld
            writer = writer_class(data=unmixed_trace_validation,
                                  **output_kwargs)
            writer.run()
            del writer
            del writer_class

        # remove invalid unmixed traces
        invalid_unmixed_trace = []
        for roi_id in unmixed_trace_validation:
            if not unmixed_trace_validation[roi_id]:
                invalid_unmixed_trace.append(roi_id)
                unmixed_traces['roi'].pop(roi_id)
                unmixed_traces['neuropil'].pop(roi_id)
        final_output[unmixed_key] = invalid_unmixed_trace

        if len(unmixed_traces['roi']) == 0:
            msg = 'No unmixed traces were valid when applying '
            msg += 'decrosstalk to ophys_experiment_id: '
            msg += '%d (%d)' % (self.experiment_id,
                                other_plane.experiment_id)
            logger.error(msg)
            return final_output

        ###################################################
        # Detect activity in unmixed traces

        unmixed_trace_events = self.get_trace_events(unmixed_traces['roi'])

        # Sometimes, unmixed_trace_events will return an array of NaNs.
        # Until we can debug that behavior, we will log those errors,
        # store the relevaten ROIs as decrosstalk_invalid_unmixed_trace,
        # and cull those ROIs from the data

        invalid_active_trace = {}
        invalid_active_trace['signal'] = []
        invalid_active_trace['crosstalk'] = []
        for roi_id in unmixed_trace_events.keys():
            local_traces = unmixed_trace_events[roi_id]
            for channel in ('signal', 'crosstalk'):
                nan_trace = np.isnan(local_traces[channel]['trace']).any()
                nan_events = np.isnan(local_traces[channel]['events']).any()
                if nan_trace or nan_events:
                    invalid_active_trace[channel].append(roi_id)

        if cache_dir is not None:
            if self.new_style_output:
                writer_class = io_utils.OutATH5Writer
            else:
                writer_class = io_utils.OutATH5WriterOld
            writer = writer_class(data=unmixed_trace_events,
                                  **output_kwargs)
            writer.run()
            del writer
            del writer_class

        n_sig = len(invalid_active_trace['signal'])
        n_ct = len(invalid_active_trace['crosstalk'])
        if n_sig > 0 or n_ct > 0:
            msg = 'ophys_experiment_id: %d (%d) ' % (self.experiment_id,
                                                     other_plane.experiment_id)
            msg += 'had ROIs with active event channels that contained NaNs'
            logger.error(msg)

            # remove ROIs with NaNs in their independent signal events
            # from the data being processed
            for roi_id in invalid_active_trace['signal']:
                final_output[unmixed_key].append(roi_id)
                unmixed_trace_events.pop(roi_id)
                unmixed_traces['roi'].pop(roi_id)
                unmixed_traces['neuropil'].pop(roi_id)

        if cache_dir is not None:
            writer = io_utils.InvalidATJsonWriter(data=invalid_active_trace,
                                                  **output_kwargs)
            writer.run()
            del writer

        unmixed_ct_ratio = self.get_crosstalk_data(unmixed_traces['roi'],
                                                   unmixed_trace_events)

        ########################################################
        # For each ROI, assess whether or not it is a "ghost"
        # (i.e. whether any of its activity is due to the signal,
        # independent of the crosstalk; if not, it is a ghost)

        independent_events = {}
        ghost_roi_id = []
        for roi_id in unmixed_trace_events:
            signal = unmixed_trace_events[roi_id]['signal']
            crosstalk = unmixed_trace_events[roi_id]['crosstalk']

            (is_a_cell,
             ind_events) = d_utils.validate_cell_crosstalk(signal, crosstalk)

            local = {'is_a_cell': is_a_cell,
                     'independent_events': ind_events}

            independent_events[roi_id] = local
            if not is_a_cell:
                ghost_roi_id.append(roi_id)
        final_output[ghost_key] = ghost_roi_id

        if cache_dir is not None:
            if self.new_style_output:
                writer_class = io_utils.ValidCTH5Writer
            else:
                writer_class = io_utils.ValidCTH5WriterOld
            writer = writer_class(data=independent_events,
                                  **output_kwargs)
            writer.run()
            del writer
            del writer_class

            crosstalk_ratio = {}
            for roi_id in unmixed_ct_ratio:
                _out = {'raw': raw_trace_crosstalk_ratio[roi_id],
                        'unmixed': unmixed_ct_ratio[roi_id]}
                crosstalk_ratio[roi_id] = _out

            if self.new_style_output:
                writer_class = io_utils.CrosstalkJsonWriter
            else:
                writer_class = io_utils.CrosstalkJsonWriterOld
            writer = writer_class(data=crosstalk_ratio, **output_kwargs)
            writer.run()
            del writer
            del writer_class

        return final_output

    def get_raw_traces(self, other_plane):
        """
        Get the raw signal and crosstalk traces comparing
        this plane to another plane

        Parameters
        ----------
        other_plane -- another instance of OphysPlane which will
        be taken as the source of crosstalk for this plane

        Returns
        -------
        A dict of raw traces such that
            output['roi'][roi_id]['signal'] is the raw signal trace for ROI

            output['roi'][roi_id]['crosstalk'] is the raw crosstalk
                                               trace for ROI

            output['neuropil'][roi_od]['signal'] is the raw signal trace for
                                                 the neuropil around ROI

            output['neuropil'][roi_id]['crosstalk'] is the raw crosstalk trace
                                                    for the neuropil around ROI
        """

        signal_traces = self.movie.get_trace(self.roi_list)
        crosstalk_traces = other_plane.movie.get_trace(self.roi_list)

        output = {}
        output['roi'] = {}
        output['neuropil'] = {}
        for roi_id in signal_traces['roi'].keys():
            _roi = {}
            _neuropil = {}

            _roi['signal'] = signal_traces['roi'][roi_id]
            _roi['crosstalk'] = crosstalk_traces['roi'][roi_id]
            output['roi'][roi_id] = _roi

            _neuropil['signal'] = signal_traces['neuropil'][roi_id]
            _neuropil['crosstalk'] = crosstalk_traces['neuropil'][roi_id]
            output['neuropil'][roi_id] = _neuropil

        return output

    def unmix_ROI(self, roi_traces, seed=None, iters=10):
        """
        Unmix the signal and crosstalk traces for a single ROI

        Parameters
        ----------
        roi_traces is a dict that such that
            roi_traces['signal'] is a numpy array
                                 containing the signal trace
            roi_traces['crosstalk'] is a numpy array
                                    containing the crosstalk trace

        seed is an int used to seed the random number generator
        that sklearn.decompositions.FastICA uses

        iters is an int indicating the number of iterations of
        FastICA to run before giving up on convegence.

        Returns
        -------
        A dict such that
            output['mixing_matrix'] -- the mixing matrix that transforms
                                       the unmixed signals back into the data
            output['signal'] -- is the unmixed signal
            output['crosstalk'] -- is the unmixed crosstalk
            output['use_avg_mixing_matrix'] -- a boolean; if True, ICA
                                               did not actually converge;
                                               we must discard these results
                                               and unmix the signal and
                                               crosstalk using the average
                                               mixing matrix for the plane
        """

        ica_input = np.array([roi_traces['signal'], roi_traces['crosstalk']])
        assert ica_input.shape == (2, len(roi_traces['signal']))

        (unmixed_signals,
         mixing_matrix,
         roi_demixed) = ica_utils.run_ica(ica_input,
                                          seed=seed,
                                          iters=iters)

        assert unmixed_signals.shape == ica_input.shape

        output = {}
        output['mixing_matrix'] = mixing_matrix
        output['signal'] = unmixed_signals[0, :]
        output['crosstalk'] = unmixed_signals[1, :]
        output['use_avg_mixing_matrix'] = not roi_demixed

        return output

    def unmix_all_ROIs(self, raw_roi_traces, seed_lookup=None):
        """
        Unmix all of the ROIs in this OphysPlane.

        Parameters
        ----------
        raw_roi_traces is a dict
            raw_roi_traces['roi'][roi_id]['signal'] is the raw signal
                                                    for the ROI

            raw_roi_traces['roi'][roi_id]['crosstalk'] is the raw crosstalk
                                                       for the ROI

            raw_roi_traces['neuropil'][roi_id]['signal'] is the raw signal
                                                   for the neuropil around
                                                   the ROI

            raw_roi_traces['neuropil'][roi_id]['crosstalk'] is the raw
                                                   crosstalk for the
                                                   neuropil around the ROI

        seed_lookup is a dict that maps roi_id to a seed for np.RandomState

        Returns
        -------
        A dict such that

            output['roi'][roi_id]['mixing_matrix'] -- the ROI's mixing matrix
            output['roi'][roi_id]['signal'] -- the ROI's unmixed signal
            output['roi'][roi_id]['crosstalk'] -- the ROI's unmixed crosstalk
            output['roi'][roi_id]['use_avg_mixing_matrix'] -- a boolean

                    If True, the ROI was demixed using the average mixing
                    matrix for the OphysPlane. In that case, the unconverged
                    mixing_matrix, signal, and crosstalk will be stored in
                    'poorly_converged_mixing_matrix', 'poorly_converged_signal'
                    and 'poorly_converged_crosstalk'

            output['neuropil'][roi_id]['signal'] -- neuropil's unmixed signal
            output['neuropil'][roi_id]['crosstalk'] -- neuropil's unmixed
                                                       crosstalk
        """

        output = {}
        output['roi'] = {}
        output['neuropil'] = {}

        # first pass naively unmixing ROIs with ICA
        for roi_id in raw_roi_traces['roi'].keys():

            unmixed_roi = self.unmix_ROI(raw_roi_traces['roi'][roi_id],
                                         seed=seed_lookup[roi_id],
                                         iters=10)

            _out = {}
            if not unmixed_roi['use_avg_mixing_matrix']:
                _out = unmixed_roi
            else:
                _out = {'use_avg_mixing_matrix': True}
                for k in unmixed_roi.keys():
                    if k == 'use_avg_mixing_matrix':
                        continue
                    _out['poorly_converged_%s' % k] = unmixed_roi[k]
            output['roi'][roi_id] = _out

        # calculate avg mixing matrix from successful iterations
        _out = output['roi']
        alpha_arr = np.array([min(_out[roi_id]['mixing_matrix'][0, 0],
                                  _out[roi_id]['mixing_matrix'][0, 1])
                              for roi_id in _out.keys()
                              if not _out[roi_id]['use_avg_mixing_matrix']])
        beta_arr = np.array([min(_out[roi_id]['mixing_matrix'][1, 0],
                                 _out[roi_id]['mixing_matrix'][1, 1])
                             for roi_id in _out.keys()
                             if not _out[roi_id]['use_avg_mixing_matrix']])

        assert alpha_arr.shape == beta_arr.shape
        if len(alpha_arr) == 0:
            return False, output

        mean_alpha = alpha_arr.mean()
        mean_beta = beta_arr.mean()
        mean_mixing_matrix = np.zeros((2, 2), dtype=float)
        mean_mixing_matrix[0, 0] = 1.0-mean_alpha
        mean_mixing_matrix[0, 1] = mean_alpha
        mean_mixing_matrix[1, 0] = mean_beta
        mean_mixing_matrix[1, 1] = 1.0-mean_beta
        inv_mean_mixing_matrix = np.linalg.inv(mean_mixing_matrix)

        for roi_id in raw_roi_traces['roi'].keys():
            inv_mixing_matrix = None
            mixing_matrix = None
            if not output['roi'][roi_id]['use_avg_mixing_matrix']:
                mixing_matrix = output['roi'][roi_id]['mixing_matrix']
                inv_mixing_matrix = np.linalg.inv(mixing_matrix)
            else:
                mixing_matrix = mean_mixing_matrix
                inv_mixing_matrix = inv_mean_mixing_matrix

                # assign missing outputs to ROIs that failed to converge
                output['roi'][roi_id]['mixing_matrix'] = mixing_matrix
                _roi_traces = raw_roi_traces['roi'][roi_id]
                unmixed_signals = np.dot(inv_mixing_matrix,
                                         np.array([_roi_traces['signal'],
                                                   _roi_traces['crosstalk']]))
                output['roi'][roi_id]['signal'] = unmixed_signals[0, :]
                output['roi'][roi_id]['crosstalk'] = unmixed_signals[1, :]

            # assign outputs to 'neuropils'
            _np_traces = raw_roi_traces['neuropil'][roi_id]
            unmixed_signals = np.dot(inv_mixing_matrix,
                                     np.array([_np_traces['signal'],
                                               _np_traces['crosstalk']]))

            output['neuropil'][roi_id] = {}
            output['neuropil'][roi_id]['signal'] = unmixed_signals[0, :]
            output['neuropil'][roi_id]['crosstalk'] = unmixed_signals[1, :]

        return True, output

    def get_trace_events(self, trace_dict):
        """
        trace_dict is a dict such that
            trace_dict[roi_id]['signal'] is the signal channel
            trace_dict[roi_id]['crosstalk'] is the crosstalk channel

        Returns
        -------
        out_dict such that
            out_dict[roi_id]['signal']['trace'] is the trace of
                                                the signal channel events

            out_dict[roi_id]['signal']['events'] is the timestep events of
                                                 the signal channel events

            out_dict[roi_id]['crosstalk']['trace'] is the trace of
                                                   the signal channel events

            out_dict[roi_id]['crosstalk']['events'] is the timestep events of
                                                    the signal channel events
        """
        roi_id_list = list(trace_dict.keys())

        data_arr = np.array([trace_dict[roi_id]['signal']
                             for roi_id in roi_id_list])
        sig_dict = active_traces.get_trace_events(data_arr,
                                                  self.trace_threshold_params)

        data_arr = np.array([trace_dict[roi_id]['crosstalk']
                             for roi_id in roi_id_list])
        ct_dict = active_traces.get_trace_events(data_arr,
                                                 self.trace_threshold_params)

        output = {}
        for i_roi, roi_id in enumerate(roi_id_list):
            local_dict = {}
            local_dict['signal'] = {}
            local_dict['crosstalk'] = {}
            local_dict['signal']['trace'] = sig_dict['trace'][i_roi]
            local_dict['signal']['events'] = sig_dict['events'][i_roi]
            local_dict['crosstalk']['trace'] = ct_dict['trace'][i_roi]
            local_dict['crosstalk']['events'] = ct_dict['events'][i_roi]
            output[roi_id] = local_dict

        return output

    def get_crosstalk_data(self, trace_dict, events_dict):
        """
        trace_dict that contains
            trace_dict[roi_id]['signal']
            trace_dict[roi_id]['crosstalk']

        events_dict that contains
            events_dict[roi_id]['signal']['trace']
            events_dict[roi_id]['signal']['events']
            events_dict[roi_id]['crosstalk']['trace']
            events_dict[roi_id]['crosstalk']['events']

        returns a dict keyed on roi_id with 100*slope relating
        signal to crosstalk
        """
        output = {}
        for roi_id in trace_dict.keys():
            signal = events_dict[roi_id]['signal']['trace']
            full_crosstalk = trace_dict[roi_id]['crosstalk']
            crosstalk = full_crosstalk[events_dict[roi_id]['signal']['events']]
            results = d_utils.get_crosstalk_data(signal, crosstalk)
            output[roi_id] = 100*results['slope']
        return output
