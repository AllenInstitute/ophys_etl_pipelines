from typing import Tuple, Optional
import numpy as np
import pathlib
import tempfile
import imageio
from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI


class ThumbnailVideo(object):
    """
    Container class to carry around the metadata describing
    a thumbnail video.

    Parameters
    ----------
    video_path: PathlibPath
        The path to the video file

    origin: Tuple[int, int]
        (row_min, col_min)

    frame_shape: Tuple[int, int]
        (n_rows, n_cols)

    timesteps: Optional[np.ndarray]
        The timesteps from the original movie that were used
        (if None, use all timesteps)

    Notes
    -----
    When the instantiation is deleted, the file containing
    the movie is also deleted
    """

    def __init__(self,
                 video_path: pathlib.Path,
                 origin: Tuple[int, int],
                 frame_shape: Tuple[int, int],
                 timesteps: Optional[np.ndarray] = None):
        self._path = video_path
        self._origin = origin
        self._frame_shape = frame_shape
        self._timesteps = timesteps

    def __del__(self):
        if self.video_path.is_file():
            self.video_path.unlink()

    @property
    def video_path(self) -> pathlib.Path:
        return self._path

    @property
    def origin(self) -> Tuple[int, int]:
        return self._origin

    @property
    def frame_shape(self) -> Tuple[int, int]:
        return self._frame_shape

    @property
    def timesteps(self) -> Optional[np.ndarray]:
        return self._timesteps


def thumbnail_video_from_array(
        full_video: np.ndarray,
        origin: Tuple[int, int],
        frame_shape: Tuple[int, int],
        timesteps: Optional[np.ndarray] = None,
        file_path: Optional[pathlib.Path] = None,
        tmp_dir: Optional[pathlib.Path] = None,
        fps: int = 31,
        quality: int = 5,
        origin_offset: Optional[Tuple[int,int]] = None) -> ThumbnailVideo:
    """
    Create a ThumbnailVideo (mp4) from a numpy array

    Parameters
    ----------
    full_video: np.ndarray
        Shape is (n_time, n_rows, n_cols)
        or (n_time, n_rows, n_cols, 3) for RGB

    origin: Tuple[int, int]
        (row_min, col_min)

    frame_shape: Tuple[int, int]
        (n_rows, n_cols)

    timesteps: Optional[np.ndarray]
        Array of timesteps. If None, keep all timesteps from
        full_video (default: None)

    file_path: Optional[pathlib.Path]
        Where to write the thumbnail video (if None, tempfile
        will be used to create a path; default is None)

    tmp_dir: Optional[pathlib.Path]
        Directory where file will be written (ignored if file_path is
        not None). If none, tempfile will be used to create a temporary
        directory (default: None)

    fps: int
        frames per second (default: 31)

    quality: int
        Parameter passed to imageio.mimsave controlling
        quality of video file produced (max is 10; default is 5)

    origin_offset: Optional[Tuple[int, int]]
        Offset values to be added to origin in container.
        *Should only be used by methods which call this method
        after pre-truncating the video in space; do NOT use this
        by hand*

    Returns
    -------
    ThumbnailVideo
        Containing the metadata about the written thumbnail video
    """

    if file_path is None:
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        file_path = tempfile.mkstemp(dir=str(tmp_dir),
                                     prefix='thumbnail_video_',
                                     suffix='.mp4')[1]
        file_path = pathlib.Path(file_path)

    if timesteps is not None:
        sub_video = full_video[timesteps]
    else:
        sub_video = full_video
    sub_video = sub_video[:,
                          origin[0]:origin[0]+frame_shape[0],
                          origin[1]:origin[1]+frame_shape[1]]

    imageio.mimsave(file_path,
                     sub_video,
                     fps=fps,
                     quality=quality,
                     pixelformat='yuvj444p')

    if origin_offset is None:
        origin_offset = (0, 0)

    container = ThumbnailVideo(file_path,
                               (origin[0]+origin_offset[0],
                                origin[1]+origin_offset[1]),
                               frame_shape,
                               timesteps=timesteps)
    return container


def thumbnail_video_from_ROI(
        full_video: np.ndarray,
        roi: ExtractROI,
        roi_color: Optional[Tuple[int, int, int]]=None,
        timesteps: Optional[np.ndarray] = None,
        file_path: Optional[pathlib.Path] = None,
        tmp_dir: Optional[pathlib.Path] = None,
        fps: int = 31,
        quality: int = 5) -> ThumbnailVideo:

    # construct an ROI object to get the boundary mask
    # for us
    roi = OphysROI(roi_id=-1,
                   x0=roi['x'],
                   y0=roi['y'],
                   width=roi['width'],
                   height=roi['height'],
                   valid_roi=False,
                   mask_matrix=roi['mask'])

    # find bounds of thumbnail

    # make thumbnail a square about the ROI
    max_dim = max(roi.width, roi.height)

    # make dim a power of 2
    pwr = np.ceil(np.log2(max_dim))
    max_dim = np.power(2, pwr).astype(int)
    max_dim = max(max_dim, 16)

    # center the thumbnail on the ROI
    row_center = int(roi.y0 + roi.height//2)
    col_center = int(roi.x0 + roi.width//2)

    rowmin = max(0, row_center - max_dim//2)
    rowmax = rowmin + max_dim
    colmin = max(0, col_center - max_dim//2)
    colmax = colmin + max_dim

    img_shape = full_video.shape[1:3]

    if rowmax >= img_shape[0]:
        rowmin = max(0, img_shape[0]-max_dim)
        rowmax = min(img_shape[0], rowmin+max_dim)
    if colmax >= img_shape[1]:
        colmin = max(0, img_shape[1]-max_dim)
        colmax = min(img_shape[1], colmin+max_dim)

    # truncate the video in time and space
    is_rgb = (len(full_video.shape) == 4)

    if timesteps is not None:
        sub_video = full_video[timesteps]
    else:
        sub_video = full_video

    sub_video = sub_video[:, rowmin:rowmax, colmin:colmax]

    if not is_rgb:
        rgb_video = np.zeros((sub_video.shape[0],
                              rowmax-rowmin,
                              colmax-colmin,
                              3), dtype=full_video.dtype)

        for ic in range(3):
            rgb_video[:, :, :, ic] = sub_video
        sub_video = rgb_video

    # if an ROI color has been specified, plot the ROI
    # boundary over the video in the specified color
    if roi_color is not None:
        boundary_mask = roi.boundary_mask
        for irow in range(boundary_mask.shape[0]):
            row = irow+roi.y0-rowmin
            for icol in range(boundary_mask.shape[1]):
                if not boundary_mask[irow, icol]:
                    continue
                col = icol+roi.x0-colmin
                for i_color in range(3):
                    sub_video[:, row, col, i_color] = roi_color[i_color]

    thumbnail = thumbnail_video_from_array(
                    sub_video,
                    (0, 0),
                    sub_video.shape[1:3],
                    timesteps=None,
                    file_path=file_path,
                    tmp_dir=tmp_dir,
                    fps=fps,
                    quality=quality,
                    origin_offset=(rowmin, colmin))

    return thumbnail
