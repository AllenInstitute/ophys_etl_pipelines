from typing import Union, Optional, Tuple

import matplotlib.figure

import copy
import h5py
import pathlib
import numpy as np

from ophys_etl.modules.segmentation.qc_utils.video_generator import (
    VideoGenerator)

from ophys_etl.modules.segmentation.qc_utils.video_utils import (
    ThumbnailVideo,
    video_bounds_from_ROI,
    add_roi_boundary_to_video)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    deserialize_extract_roi_list)

import json


def get_roi_list_in_fov(roi_list, origin, frame_shape):
    global_r0 = origin[0]
    global_r1 = global_r0 + frame_shape[0]
    global_c0 = origin[1]
    global_c1 = global_c0 + frame_shape[1]

    output = []
    for roi in roi_list:
        r0 = roi['y']
        r1 = r0+roi['height']
        c0 = roi['x']
        c1 = c0+roi['width']
        if r1 < global_r0:
            continue
        elif r0 > global_r1:
            continue
        elif c1 < global_c0:
            continue
        elif c0 > global_c1:
            continue

        output.append(roi)
    return output


class Classifier_ROISet(object):
    """
    Parameters
    ----------
    artifact_path: Union[str, pathlib.Path]
        Path to the HDF5 file containing the precomputed artifact data

    tmp_dir: Union[None, str, pathlib.Path]
        Path to the temporary directory where video files will get written
        (default: None)
    """

    def __init__(
            self,
            artifact_path: Union[str, pathlib.Path],
            tmp_dir: Union[None, str, pathlib.Path] = None):

        if isinstance(artifact_path, str):
            artifact_path = pathlib.Path(artifact_path)

        if not artifact_path.is_file():
            msg = f'\n{artifact_path}\nis not a file'
            raise RuntimeError(msg)

        if isinstance(tmp_dir, str):
            tmp_dir = pathlib.Path(tmp_dir)

        with h5py.File(artifact_path, 'r') as in_file:
            raw_color_map = json.loads(
                               in_file['roi_color_map'][()].decode('utf-8'))
            self.color_map = {int(roi_id): tuple(raw_color_map[roi_id])
                              for roi_id in raw_color_map}
            self.extract_roi_list = deserialize_extract_roi_list(
                                            in_file['rois'][()])
            self.extract_roi_lookup = {roi['id']: roi
                                       for roi in self.extract_roi_list}

            trace_group = in_file['traces']
            self.trace_lookup = {int(roi_id): trace_group[roi_id][()]
                                 for roi_id in trace_group.keys()}

            self.video_generator = VideoGenerator(
                                       video_data=in_file['video_data'][()],
                                       tmp_dir=tmp_dir)

            raw_max_projection = in_file['max_projection'][()]


        self.max_projection = np.zeros((raw_max_projection.shape[0],
                                        raw_max_projection.shape[1],
                                        3), dtype=np.uint8)

        for ic in range(3):
            self.max_projection[:, :, ic] = raw_max_projection


    def get_roi(self, roi_id: int):
        roi = Classifier_ROI(roi_id, self)
        return roi


    def mark_roi_valid(self, roi_id: int):
        if roi_id not in self.extract_roi_lookup:
            raise RuntimeError(f"{roi_id} is not a valid ROI ID")
        self.extract_roi_lookup[roi_id]['valid'] = True

    def mark_roi_invalid(self, roi_id:int):
        if roi_id not in self.extract_roi_lookup:
            raise RuntimeError(f"{roi_id} is not a valid ROI ID")
        self.extract_roi_lookup[roi_id]['valid'] = False

    def get_roi_video(
            self,
            roi_id: int,
            roi_color: Optional[Tuple[int, int, int]]=None,
            other_rois: bool = False,
            padding: int = 8,
            timesteps: Optional[np.ndarray] = None) -> ThumbnailVideo:

        this_roi = self.extract_roi_lookup[roi_id]

        if other_rois:
            if roi_color is None:
                raise RuntimeError("other_rois is True, "
                                   "but roi_color is None")
            this_color_map = copy.deepcopy(self.color_map)
            this_color_map[roi_id] = roi_color

            roi_list = self.extract_roi_list
        else:
            roi_list = None
            this_color_map = roi_color
        video = self.video_generator.get_thumbnail_video_from_roi(
                        this_roi,
                        padding=padding,
                        quality=9,
                        timesteps=timesteps,
                        fps=31,
                        other_roi=roi_list,
                        roi_color=this_color_map)
        return video


    def get_trace_plot(
            self,
            roi_id: int,
            timesteps: Optional[np.ndarray]=None) -> matplotlib.figure.Figure:
        trace = self.trace_lookup[roi_id]
        fig = matplotlib.figure.Figure(figsize=(15, 5))
        axis = fig.add_subplot(1,1,1)
        if timesteps is None:
            timesteps = np.arange(len(trace), dtype=int)
        axis.plot(timesteps, trace[timesteps])
        fig.tight_layout()
        return fig


    def get_max_projection_plot(
            self,
            roi_id: int,
            roi_color: Tuple[int, int, int] = (255, 0, 0),
            other_rois: bool = False,
            padding:int = 32) -> matplotlib.figure.Figure:

        this_roi = self.extract_roi_lookup[roi_id]

        (origin,
         frame_shape) = video_bounds_from_ROI(
                               this_roi,
                               self.max_projection.shape,
                               padding)

        # cast as array with 3-axes so that we can use the
        # add_roi_boundary_to_video method, which already is designed
        # to handle subsected videos/images
        global_r0 = origin[0]
        global_r1 = global_r0 + frame_shape[0]
        global_c0 = origin[1]
        global_c1 = global_c0 + frame_shape[1]

        img = np.array([np.copy(self.max_projection[global_r0:global_r1,
                                                    global_c0:global_c1])])

        if other_rois:
            this_color_map = copy.deepcopy(self.color_map)
            roi_list = self.extract_roi_list
        else:
            this_color_map = dict()
            roi_list = [this_roi]
        this_color_map[this_roi['id']] = roi_color


        for roi in get_roi_list_in_fov(roi_list, origin, img.shape[1:]):
            this_color = this_color_map[roi['id']]
            if this_color is None:
                continue
            img = add_roi_boundary_to_video(
                       img,
                       origin,
                       roi,
                       this_color)

        img = img[0, :, :]
        width = img.shape[1]
        height = img.shape[0]
        ratio = width/height

        fig_height = 10
        fig = matplotlib.figure.Figure(figsize=(np.round(fig_height*ratio).astype(int),
                                                 fig_height))
        axis = fig.add_subplot(1,1,1)
        axis.imshow(img)
        fig.tight_layout()
        return fig


    def get_summary_figure(
            self,
            origin: Optional[Tuple[int, int]] = None,
            frame_shape: Optional[Tuple[int, int]] = None) -> matplotlib.figure.Figure:
        """
        Will only plot ROIs that are marked as valid in self.extract_roi_lookup
        """
        if origin is None:
            origin = (0, 0)
        if frame_shape is None:
            frame_shape = self.max_projection.shape

        global_r0 = max(0, origin[0])
        global_r1 = min(self.max_projection.shape[0], global_r0+frame_shape[0])
        global_c0 = max(0, origin[1])
        global_c1 = min(self.max_projection.shape[1], global_c0+frame_shape[1])

        img = np.array([np.copy(self.max_projection[global_r0:global_r1,
                                                    global_c0:global_c1])])
        roi_list = [roi for roi in self.extract_roi_lookup.values()
                    if roi['valid']]

        for roi in get_roi_list_in_fov(roi_list, origin, img.shape[1:]):
            img = add_roi_boundary_to_video(
                      img,
                      origin,
                      roi,
                      self.color_map[roi['id']])

        fig = matplotlib.figure.Figure(figsize=(20, 20))
        axis = fig.add_subplot(1,1,1)
        axis.imshow(img[0, :, :])
        fig.tight_layout()
        return fig


class Classifier_ROI(object):
    """
    Should not be instantiated by hand;
    meant to be spun off from a Classifier_ROISet
    """

    def __init__(self,
                 roi_id: int,
                 roi_set: Classifier_ROISet):
        self.roi_id = roi_id
        self.roi_set = roi_set

    def mark_valid(self):
        self.roi_set.mark_roi_valid(self.roi_id)

    def mark_invalid(self):
        self.roi_set.mark_roi_invalid(self.roi_id)

    def get_trace_plot(self,
                       timesteps: Optional[np.ndarray]=None):
        return self.roi_set.get_trace_plot(
                    roi_id=self.roi_id,
                    timesteps=timesteps)

    def get_thumbnail_video(self,
                            padding: int = 32,
                            include_boundary: bool = True,
                            include_others: bool = False,
                            timesteps: Optional[np.ndarray] = None) -> ThumbnailVideo:

        if include_boundary:
            roi_color = (255, 0, 0)
        else:
            roi_color = None
            if include_others:
                raise RuntimeError("You just asked to include all "
                                   "ROIs *except* this one in your video")

        return self.roi_set.get_roi_video(
                        self.roi_id,
                        roi_color=roi_color,
                        other_rois=include_others,
                        timesteps=timesteps,
                        padding=padding)

    def get_max_projection(self,
                           padding: int = 32,
                           include_boundary: bool = True,
                           include_others: bool = False) -> matplotlib.figure.Figure:

        if include_boundary:
            roi_color = (255, 0, 0)
        else:
            roi_color = None
            if include_others:
                raise RuntimeError("You just asked to include all "
                                   "ROIs *except* this one in your image")

        return self.roi_set.get_max_projection_plot(
                        self.roi_id,
                        roi_color = roi_color,
                        other_rois = include_others,
                        padding=padding)
