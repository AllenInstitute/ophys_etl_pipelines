from typing import Optional, List, Tuple, Dict, Union
import pathlib
import numpy as np
import h5py
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
         quality: int = 5,
         fps: int = 31,
         tmp_dir: Optional[pathlib.Path] = None):

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

    new_roi = ExtractROI(
                   id=int(roi['id']),
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
