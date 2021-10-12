from typing import Union, Optional, Tuple

import matplotlib.figure

import copy
import h5py
import pathlib
import numpy as np

from ophys_etl.modules.segmentation.qc_utils.video_generator import (
    VideoGenerator)

from ophys_etl.modules.segmentation.qc_utils.video_utils import (
    ThumbnailVideo)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    deserialize_extract_roi_list)

import json


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
            self.max_projection = in_file['max_projection'][()]
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
        return fig
