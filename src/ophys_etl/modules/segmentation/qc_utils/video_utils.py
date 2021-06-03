from typing import Tuple, Optional
import numpy as np
import pathlib
import tempfile
import imageio


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
        fps: int = 31) -> ThumbnailVideo:
    """
    Create a ThumbnailVideo (mp4) from a numpy array

    Parameters
    ----------
    full_video: np.ndarray
        Shape is (n_time, n_rows, n_cols)

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
        sub_video = full_video[timesteps, :, :]
    else:
        sub_video = full_video
    sub_video = sub_video[:,
                          origin[0]:origin[0]+frame_shape[0],
                          origin[1]:origin[1]+frame_shape[1]]

    imageio.mimsave(file_path,
                     sub_video,
                     fps=fps,
                     quality=10)

    container = ThumbnailVideo(file_path,
                               origin,
                               frame_shape,
                               timesteps=timesteps)
    return container
