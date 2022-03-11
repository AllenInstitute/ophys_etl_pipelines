from typing import Optional, List, Tuple, Dict, Union
import pathlib
import numpy as np
import h5py
import numbers
from ophys_etl.types import ExtractROI
from ophys_etl.utils.video_utils import video_bounds_from_ROI
import ophys_etl.utils.thumbnail_video_utils as thumbnail_utils


def get_thumbnail_video_from_artifact_file(
         artifact_path: pathlib.Path,
         roi: ExtractROI,
         padding: int = 0,
         other_roi: Union[None, List[ExtractROI]] = None,
         roi_color: Union[None,
                          Tuple[int, int, int],
                          Dict[int, Tuple[int, int, int]]] = None,
         timesteps: Optional[np.ndarray] = None,
         fps: int = 31,
         quality: int = 5,
         tmp_dir: Optional[pathlib.Path] = None
         ) -> thumbnail_utils.ThumbnailVideo:
    """
    Get a ThumbnailVideo from an ROI and a labeler artifact file

    Parameters
    ----------
    artifact_path: pathlib.Path
        Path to the labeler artifact file from which to read the video

    roi: ExtractROI

    padding: int
        The number of pixels to either side of the ROI to
        include in the field of view (if possible; default=0)

    other_roi: Union[None, List[ExtractROI]]
        Other ROI to display

    roi_color: Union[None,
                     Tuple[int, int, int],
                     Dict[int, Tuple[int, int, int]]]
        If not None, the RGB color in which to plot the ROI's
        contour (or dict mapping ROI ID to RGB color).
        If None, ROI is not plotted in thumbnail.
        (default: None)

    timesteps: Optional[np.ndarray]
        If not None, timesteps to put in the thumbnail
        video. If None, use all timesetps (default: None)

    fps: int
        frames per second (default: 31)

    quality: int
        quality parameter passed to imageio. Max is 10.
        (default: 5)

    tmp_dir: Optional[pathlib.Path]
        temporary directory where thumbnail video will
        be written

    Returns
    -------
    thumbnail_utils.ThumbnailVideo
    """

    with h5py.File(artifact_path, 'r') as in_file:
        fov_shape = in_file['video_data'].shape[1:]
        (origin,
         window_shape) = video_bounds_from_ROI(
                              roi=roi,
                              fov_shape=fov_shape,
                              padding=padding)

        y0 = origin[0]
        y1 = origin[0]+window_shape[0]
        x0 = origin[1]
        x1 = origin[1]+window_shape[1]

        if timesteps is None:
            video_data = in_file['video_data'][:, y0:y1, x0:x1]
        else:
            video_data = in_file['video_data'][timesteps, y0:y1, x0:x1]

    # When users of the cell labeling app try to load a video from
    # an arbitrary point, the assigned id is a string, not an int.
    # thumbnail_video_from_ROI() below will not handle that well
    if not isinstance(roi['id'], numbers.Number):
        new_id = -999
        if isinstance(roi_color, dict):
            roi_color = copy.deepcopy(roi_color)
            roi_color[new_id] = roi_color[roi['id']]
    else:
        new_id = int(roi['id'])

    new_roi = ExtractROI(
                   id=new_id,
                   y=int(roi['y']-y0),
                   x=int(roi['x']-x0),
                   width=int(roi['width']),
                   height=int(roi['height']),
                   mask=roi['mask'],
                   valid=True)
    if other_roi is None:
        new_other_roi = None
    else:
        new_other_roi = []
        for o_roi in other_roi:
            new_o_roi = ExtractROI(
                           id=int(o_roi['id']),
                           y=int(o_roi['y']-y0),
                           x=int(o_roi['x']-x0),
                           width=int(o_roi['width']),
                           height=int(o_roi['height']),
                           mask=o_roi['mask'],
                           valid=True)
            new_other_roi.append(new_o_roi)

    thumbnail = thumbnail_utils.thumbnail_video_from_ROI(
                    video=video_data,
                    roi=new_roi,
                    padding=padding,
                    roi_color=roi_color,
                    other_roi=new_other_roi,
                    timesteps=None,
                    tmp_dir=tmp_dir,
                    quality=quality,
                    min_max=None,
                    fps=fps)

    return thumbnail
