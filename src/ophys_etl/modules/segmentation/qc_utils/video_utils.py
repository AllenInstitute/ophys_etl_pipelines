from typing import Tuple, Optional, Union
import numpy as np
import pathlib
import tempfile
import imageio
import h5py
import numbers
from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI


class ThumbnailVideo(object):
    """
    Container class to carry around the metadata describing
    a thumbnail video.

    Parameters
    ----------
    video_data: np.ndarray

    video_path: PathlibPath
        The path to the video file

    origin: Tuple[int, int]
        The origin used to create video_data,
        if relevant (row_min, col_min)

    timesteps: Optional[np.ndarray]
        The timesteps from the original movie that were used
        to create video_data, if relevant (default: None)

    quality: int
        Quality parameter passed to imageio.mimsave
        (maximum is 10; default is 5)

    fps: int
        frames per second; default 31

    Notes
    -----
    The constructor for this class will save video_data as
    an mp4 to a file at the specivied video_path.

    When the instantiation of this class is deleted, the mp4
    it created will be deleted.
    """

    def __init__(self,
                 video_data: np.ndarray,
                 video_path: pathlib.Path,
                 origin: Tuple[int, int],
                 timesteps: Optional[np.ndarray] = None,
                 quality: int = 5,
                 fps: int = 31):
        self._path = video_path
        self._origin = origin
        self._frame_shape = video_data.shape[1:3]
        if timesteps is None:
            self._timesteps = timesteps
        else:
            self._timesteps = np.copy(timesteps)

        imageio.mimsave(self._path,
                        video_data,
                        fps=fps,
                        quality=quality,
                        pixelformat='yuv420p',
                        codec='libx264')

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


def scale_video_to_uint8(video: np.ndarray,
                         max_val: Union[int, float]):
    """
    Convert a video (as a numpy.ndarray) to uint8 by dividing by the
    array's maximum value and multiplying by 255

    Parameters
    ----------
    video: np.ndarray

    max_val: Optional[Union[int, float]]
        The value by which to normalize before multiplying by 255

    Returns
    -------
    np.ndarray
    """
    if max_val is None:
        raise RuntimeError("must specify a max_val in "
                           "scale_video_to_uint8")

    if max_val < video.max():
        msg = "in scale_video_to_uint8, video.max() is "
        msg += f"{video.max()}; you specified "
        msg += f"max_val = {max_val}; this will result "
        msg += "in values > 255 after normalization"
        raise RuntimeError(msg)

    return np.round(255*video.astype(float)/max_val).astype(np.uint8)


def trim_video(
        full_video: np.ndarray,
        origin: Tuple[int, int],
        frame_shape: Tuple[int, int],
        timesteps: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Create a thumbnail video from a numpy array. This method
    will do the work of trimming the array in space and time.

    No attempt is made to scale or convert the data type of
    full_video's contents.

    Parameters
    ----------
    full_video: np.ndarray
        Shape is (n_time, n_rows, n_cols)
        or (n_time, n_rows, n_cols, 3) for RGB

    origin: Tuple[int, int]
        (row_min, col_min) of desired thumbnail

    frame_shape: Tuple[int, int]
        (n_rows, n_cols) of desired thumbnail

    timesteps: Optional[np.ndarray]
        Array of timesteps. If None, keep all timesteps from
        full_video (default: None)

    Returns
    -------
    np.ndarray
        Containing the thumbnail video data
    """

    if timesteps is not None:
        sub_video = full_video[timesteps]
    else:
        sub_video = full_video

    sub_video = sub_video[:,
                          origin[0]:origin[0]+frame_shape[0],
                          origin[1]:origin[1]+frame_shape[1]]

    return sub_video


def thumbnail_video_from_array(
        full_video: np.ndarray,
        origin: Tuple[int, int],
        frame_shape: Tuple[int, int],
        timesteps: Optional[np.ndarray] = None,
        file_path: Optional[pathlib.Path] = None,
        tmp_dir: Optional[pathlib.Path] = None,
        fps: int = 31,
        quality: int = 5,
        origin_offset: Optional[Tuple[int, int]] = None,
        timestep_offset: Optional[np.ndarray] = None) -> ThumbnailVideo:
    """
    Create a ThumbnailVideo (mp4) from a numpy array. This method
    will do the work of trimming the array in space and time.

    No attempt is made to scale or convert the data type of
    full_video's contents.

    Parameters
    ----------
    full_video: np.ndarray
        Shape is (n_time, n_rows, n_cols)
        or (n_time, n_rows, n_cols, 3) for RGB

    origin: Tuple[int, int]
        (row_min, col_min) of desired thumbnail

    frame_shape: Tuple[int, int]
        (n_rows, n_cols) of desired thumbnail

    timesteps: Optional[np.ndarray]
        Array of timesteps. If None, keep all timesteps from
        full_video (default: None)

    file_path: Optional[pathlib.Path]
        Where to write the thumbnail video (if None, tempfile
        will be used to create a path; default is None)

    tmp_dir: Optional[pathlib.Path]
        Directory where file will be written (ignored if file_path is
        not None). If None, tempfile will be used to create a temporary
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
        by hand.*

    timestep_offset: Optional[Tuple[int, int]]
        timestep values to be saved in output, even though the
        cut in time has already been applied to the sub_video.
        *Should only be used by methods which call this method
        after pre-truncating the video in space; do NOT use this
        by hand.*

    Returns
    -------
    ThumbnailVideo
        Containing the metadata about the written thumbnail video
    """

    if timesteps is not None and timestep_offset is not None:
        msg = "You have called thumbnail_video_from_array "
        msg += "with non-None timesteps and non-None "
        msg += "timestep_offset; only one of these can be "
        msg += "non-None at a time."
        raise RuntimeError(msg)

    if file_path is None:
        if tmp_dir is None:
            tmp_dir = tempfile.mkdtemp()
        file_path = tempfile.mkstemp(dir=str(tmp_dir),
                                     prefix='thumbnail_video_',
                                     suffix='.mp4')[1]
        file_path = pathlib.Path(file_path)

    if origin_offset is None:
        origin_offset = (0, 0)

    sub_video = trim_video(
                    full_video,
                    origin,
                    frame_shape,
                    timesteps=timesteps)

    if timesteps is None:
        if timestep_offset is not None:
            timesteps = timestep_offset

    container = ThumbnailVideo(sub_video,
                               file_path,
                               (origin[0]+origin_offset[0],
                                origin[1]+origin_offset[1]),
                               timesteps=timesteps,
                               fps=fps,
                               quality=quality)
    return container


def thumbnail_video_from_path(
        full_video_path: pathlib.Path,
        origin: Tuple[int, int],
        frame_shape: Tuple[int, int],
        timesteps: Optional[np.ndarray] = None,
        file_path: Optional[pathlib.Path] = None,
        tmp_dir: Optional[pathlib.Path] = None,
        fps: int = 31,
        quality: int = 5,
        normalization: Union[str, float, int] = 'local',
        origin_offset: Optional[Tuple[int, int]] = None) -> ThumbnailVideo:
    """
    Create a ThumbnailVideo (mp4) from a path to an HDF5 file.
    Automatically converts video to an array of np.uint8s

    Parameters
    ----------
    full_video_path: pathlib.Path
        Path to the h5 file

    origin: Tuple[int, int]
        (row_min, col_min) of the desired thumbnail

    frame_shape: Tuple[int, int]
        (n_rows, n_cols) of the desired thumbnail

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

    normalization: Union[str, float, int]
        If 'global', normalize video to maximum value of full
        video. If 'local', normalize video to maximum value of
        spatial thumbnail. If a number, the numerical
        value to normalize by (default: 'local')

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

    if not isinstance(normalization, numbers.Number):
        if normalization not in ('global', 'local'):
            msg = "normalization in thumbnail_video_from_path "
            msg += "must be either 'global' or 'local'; "
            msg += f"you gave '{normalization}'"
            raise RuntimeError(msg)
    else:
        max_val = normalization

    with h5py.File(full_video_path, 'r') as in_file:
        if normalization == 'global':
            max_val = in_file['data'][()].max()
        data = in_file['data'][:,
                               origin[0]:origin[0]+frame_shape[0],
                               origin[1]:origin[1]+frame_shape[1]]

    if normalization == 'local':
        max_val = data.max()

    data = scale_video_to_uint8(data, max_val=max_val)

    # origin is set to (0,0) because, when we read in the
    # HDF5 file, we only read in the pixels we actually
    # wanted for the thumbnail
    thumbnail = thumbnail_video_from_array(
                    data,
                    (0, 0),
                    frame_shape,
                    timesteps=timesteps,
                    file_path=file_path,
                    tmp_dir=tmp_dir,
                    fps=fps,
                    quality=quality,
                    origin_offset=origin)
    return thumbnail


def video_bounds_from_ROI(
        roi: ExtractROI,
        fov_shape: Tuple[int, int]):
    """
    Get the field of view bounds from an ROI

    Parameters
    ----------
    roi: ExtractROI

    fov_shape: Tuple[int, int]
        The 2-D shape of the full field of view

    Returns
    -------
    origin: Tuple[int, int]
        The origin of a sub field of view containing
        the ROI

    shape: Tuple[int, int]
        The shape of the sub field of view

    Notes
    -----
    Will try to return a square that is an integer
    power of 2 on a side, minimum 16
    """

    # make thumbnail a square about the ROI
    max_dim = max(roi['width'], roi['height'])

    # make dim a power of 2
    pwr = np.ceil(np.log2(max_dim))
    max_dim = np.power(2, pwr).astype(int)
    max_dim = max(max_dim, 16)

    # center the thumbnail on the ROI
    row_center = int(roi['y'] + roi['height']//2)
    col_center = int(roi['x'] + roi['width']//2)

    rowmin = max(0, row_center - max_dim//2)
    rowmax = rowmin + max_dim
    colmin = max(0, col_center - max_dim//2)
    colmax = colmin + max_dim

    if rowmax >= fov_shape[0]:
        rowmin = max(0, fov_shape[0]-max_dim)
        rowmax = min(fov_shape[0], rowmin+max_dim)
    if colmax >= fov_shape[1]:
        colmin = max(0, fov_shape[1]-max_dim)
        colmax = min(fov_shape[1], colmin+max_dim)

    return (rowmin, colmin), (rowmax-rowmin, colmax-colmin)


def add_roi_boundary_to_video(sub_video: np.ndarray,
                              origin: Tuple[int, int],
                              roi: ExtractROI,
                              roi_color: Tuple[int, int, int]) -> np.ndarray:
    """
    Add the boundary of an ROI to a video

    Parameters
    ----------
    sub_video: np.ndarray
        The video as an RGB movie. Shape is (n_t, nrows, ncols, 3)

    origin: Tuple[int, int]
        global (romin, colmin) of the video in sub_video

    roi: ExtractROI
        The parameters of this ROI are in global coordinates,
        which is why we need origin as an argument

    roi_color: Tuple[int, int, int]
        RGB color of the ROI boundary

    Returns
    -------
    sub_video_bdry: np.ndarray
        sub_video with the ROI boundary
    """

    sub_video_bdry = np.copy(sub_video)

    # construct an ROI object to get the boundary mask
    # for us
    ophys_roi = OphysROI(roi_id=-1,
                         x0=roi['x'],
                         y0=roi['y'],
                         width=roi['width'],
                         height=roi['height'],
                         valid_roi=False,
                         mask_matrix=roi['mask'])

    boundary_mask = ophys_roi.boundary_mask
    for irow in range(boundary_mask.shape[0]):
        row = irow+ophys_roi.y0-origin[0]
        for icol in range(boundary_mask.shape[1]):
            if not boundary_mask[irow, icol]:
                continue
            col = icol+ophys_roi.x0-origin[1]
            for i_color in range(3):
                sub_video_bdry[:, row, col, i_color] = roi_color[i_color]
    return sub_video_bdry


def get_rgb_sub_video(full_video: np.ndarray,
                      origin: Tuple[int, int],
                      fov_shape: Tuple[int, int],
                      timesteps: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Take a (n_times, nrows, ncols) np.ndarray and extract a
    into a (n_times, nrows, ncols, 3) thumbnail from it.

    Parameters
    ----------
    full_video: np.ndarray

    origin: Tuple[int, int]
        (rowmin, colmin) of the desired thumbnail

    fov_shape: Tuple[int, int]
        (nrows, ncols) of the desired thumbnail

    timesteps: Optional[np.ndarray]
        Timesteps of full_video to be copied into the thumbnail
        (if None, use all timesteps; default=None)

    Returns
    -------
    sub_video: np.ndarray
        The thumbnail video shaped like an RGB video
    """
    # truncate the video in time and space
    is_rgb = (len(full_video.shape) == 4)

    if timesteps is not None:
        sub_video = full_video[timesteps]
    else:
        sub_video = full_video

    rowmin = origin[0]
    rowmax = origin[0]+fov_shape[0]
    colmin = origin[1]
    colmax = origin[1]+fov_shape[1]

    sub_video = sub_video[:, rowmin:rowmax, colmin:colmax]

    if not is_rgb:
        rgb_video = np.zeros((sub_video.shape[0],
                              fov_shape[0],
                              fov_shape[1],
                              3), dtype=full_video.dtype)

        for ic in range(3):
            rgb_video[:, :, :, ic] = sub_video
        sub_video = rgb_video

    return sub_video


def _thumbnail_video_from_ROI_array(
        full_video: np.ndarray,
        roi: ExtractROI,
        roi_color: Optional[Tuple[int, int, int]] = None,
        timesteps: Optional[np.ndarray] = None,
        file_path: Optional[pathlib.Path] = None,
        tmp_dir: Optional[pathlib.Path] = None,
        fps: int = 31,
        quality: int = 5) -> ThumbnailVideo:
    """
    Get a thumbnail video from a np.ndarray and an ROI

    Parameters
    ----------
    full_video: np.ndarray
        shape is (n_times, nrows, ncols)

    roi: ExtractROI

    roi_color: Tuple[int, int, int]
        RGB color in which to draw the ROI in the video
        (if None, ROI is not drawn; default = None)

    timesteps: Optional[np.ndarray]
        Timesteps of full_video to be copied into the thumbnail
        (if None, use all timesteps; default=None)

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
        Quality parameter passed to imageio.mimsave
        (maximum is 10; default is 5)

    Returns
    -------
    thumbnail: ThumbnailVideo

    Notes
    -----
    This method will *not* do the work of scaling full_video
    values to [0, 255]
    """

    # find bounds of thumbnail
    (origin,
     fov_shape) = video_bounds_from_ROI(roi,
                                        full_video.shape[1:3])

    sub_video = get_rgb_sub_video(full_video,
                                  origin,
                                  fov_shape,
                                  timesteps=timesteps)

    # if an ROI color has been specified, plot the ROI
    # boundary over the video in the specified color
    if roi_color is not None:
        sub_video = add_roi_boundary_to_video(sub_video,
                                              origin,
                                              roi,
                                              roi_color)

    thumbnail = thumbnail_video_from_array(
                    sub_video,
                    (0, 0),
                    sub_video.shape[1:3],
                    timesteps=None,
                    file_path=file_path,
                    tmp_dir=tmp_dir,
                    fps=fps,
                    quality=quality,
                    origin_offset=origin)

    return thumbnail


def _thumbnail_video_from_ROI_path(
        video_path: pathlib.Path,
        roi: ExtractROI,
        roi_color: Optional[Tuple[int, int, int]] = None,
        timesteps: Optional[np.ndarray] = None,
        file_path: Optional[pathlib.Path] = None,
        tmp_dir: Optional[pathlib.Path] = None,
        fps: int = 31,
        quality: int = 5,
        normalization: Union[str, int, float] = 'local') -> ThumbnailVideo:
    """
    Get a thumbnail video from a HDF5 file path and an ROI

    Parameters
    ----------
    video_path: pathlib.Path
        path to HDF5 file storing video data.
        Shape of data is (n_times, nrows, ncols)

    roi: ExtractROI

    roi_color: Tuple[int, int, int]
        RGB color in which to draw the ROI in the video
        (if None, ROI is not drawn; default = None)

    timesteps: Optional[np.ndarray]
        Timesteps of full_video to be copied into the thumbnail
        (if None, use all timesteps; default=None)

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
        Quality parameter passed to imageio.mimsave
        (maximum is 10; default is 5)

    normalization: Union[str, float]
        If 'global', normalize video to maximum value of full
        video. If 'local', normalize video to maximum value of
        spatial thumbnail. If a number, the numerical
        value to normalize by (default: 'local')

    Returns
    -------
    thumbnail: ThumbnailVideo

    Notes
    -----
    This method will scale video data values to [0, 255]
    """

    if not isinstance(normalization, numbers.Number):
        if normalization not in ('global', 'local'):
            msg = "normalization in _thumbnail_video_from_ROI_path "
            msg += "must be either 'global' or 'local'; "
            msg += f"you gave '{normalization}'"
            raise RuntimeError(msg)
    else:
        max_val = normalization

    with h5py.File(video_path, 'r') as in_file:
        img_shape = in_file['data'].shape

    # find bounds of thumbnail
    (origin,
     fov_shape) = video_bounds_from_ROI(roi,
                                        img_shape[1:3])

    with h5py.File(video_path, 'r') as in_file:
        if normalization == 'global':
            max_val = in_file['data'][()].max()

        full_video = in_file['data'][:,
                                     origin[0]:origin[0]+fov_shape[0],
                                     origin[1]:origin[1]+fov_shape[1]]

    if normalization == 'local':
        max_val = full_video.max()

    full_video = scale_video_to_uint8(full_video,
                                      max_val=max_val)

    sub_video = get_rgb_sub_video(full_video,
                                  (0, 0),
                                  fov_shape,
                                  timesteps=timesteps)

    # if an ROI color has been specified, plot the ROI
    # boundary over the video in the specified color
    if roi_color is not None:
        sub_video = add_roi_boundary_to_video(sub_video,
                                              origin,
                                              roi,
                                              roi_color)

    thumbnail = thumbnail_video_from_array(
                    sub_video,
                    (0, 0),
                    sub_video.shape[1:3],
                    timesteps=None,
                    file_path=file_path,
                    tmp_dir=tmp_dir,
                    fps=fps,
                    quality=quality,
                    origin_offset=origin)

    return thumbnail


def thumbnail_video_from_ROI(
        video: Union[np.ndarray, pathlib.Path],
        roi: ExtractROI,
        roi_color: Optional[Tuple[int, int, int]] = None,
        timesteps: Optional[np.ndarray] = None,
        file_path: Optional[pathlib.Path] = None,
        tmp_dir: Optional[pathlib.Path] = None,
        fps: int = 31,
        quality: int = 5,
        normalization: Union[str, int, float] = 'local') -> ThumbnailVideo:
    """
    Get a thumbnail video from a HDF5 file path and an ROI

    Parameters
    ----------
    video: Union[np.ndarray, pathlib.Path]
        Either a np.ndarray containing video data or the path
        to an HDF5 file containing said array. In either case,
        data is assumed to be shaped like (n_times, nrows, ncols)

    roi: ExtractROI

    roi_color: Tuple[int, int, int]
        RGB color in which to draw the ROI in the video
        (if None, ROI is not drawn; default = None)

    timesteps: Optional[np.ndarray]
        Timesteps of full_video to be copied into the thumbnail
        (if None, use all timesteps; default=None)

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
        Quality parameter passed to imageio.mimsave
        (maximum is 10; default is 5)

    normalization: Union[str, float]
        If 'global', normalize video to maximum value of full
        video. If 'local', normalize video to maximum value of
        spatial thumbnail. If a number, the numerical
        value to normalize by (default: 'local')

    Returns
    -------
    thumbnail: ThumbnailVideo

    Notes
    -----
    If video is a np.ndarray, data will not be scaled so that values
    are in [0, 255]. If video is a path, the data will be scaled
    so that values are in [0, 255]
    """

    if isinstance(video, np.ndarray):
        thumbnail = _thumbnail_video_from_ROI_array(
                           video,
                           roi,
                           roi_color=roi_color,
                           timesteps=timesteps,
                           file_path=file_path,
                           tmp_dir=tmp_dir,
                           fps=fps,
                           quality=quality)
    elif isinstance(video, pathlib.Path):
        thumbnail = _thumbnail_video_from_ROI_path(
                           video,
                           roi,
                           roi_color=roi_color,
                           timesteps=timesteps,
                           file_path=file_path,
                           tmp_dir=tmp_dir,
                           fps=fps,
                           quality=quality,
                           normalization=normalization)
    else:
        msg = "video must be either a np.ndarray "
        msg += "or a pathlib.Path; you passed in "
        msg += f"{type(video)}"
        raise RuntimeError(msg)
    return thumbnail
