from typing import Tuple, Optional, Union, List, Dict
import numpy as np
import pathlib
import tempfile
import imageio
import h5py
import numbers
from ophys_etl.types import ExtractROI, OphysROI
from ophys_etl.utils.video_utils import read_and_scale
from ophys_etl.utils.video_utils import video_bounds_from_ROI
from ophys_etl.utils.rois import (
    get_roi_list_in_fov)


def upscale_video_frame(raw_video: np.ndarray,
                        upscale_factor: int) -> np.ndarray:
    """
    Increase the frame size of a video. Return the new video.

    Parameters
    ---------
    raw_video: np.ndarray
        Video data of shape (ntime, nrows, ncols) or
        (ntime, nrows, ncols)

    upscale_factor: int
        The factor by which to upscale each dimension of the video's
        frame size

    Returns
    -------
    new_video: np.ndarray
        Video produced by upscaling the dimensions of raw_video
        (if necessar) and replicating pixels in blocks to produce
        the same video.

    Notes
    -----
    This is necessary because the pixel format we are forced to use
    (yuv420p) is such that each 4-pixel block of pixels shares UV
    values, small thumbnails can come out fuzzy due to interpolation
    across pixels that should be distinct. This method works around
    that shortcoming by duplicating pixels so that each pixel in the
    input video corresponds to a contiguous block of pixels in the
    output video.
    """
    nt = raw_video.shape[0]
    nrows = raw_video.shape[1]
    ncols = raw_video.shape[2]
    if len(raw_video.shape) == 4:
        new_video = np.zeros((nt,
                              upscale_factor*nrows,
                              upscale_factor*ncols,
                              raw_video.shape[3]), dtype=raw_video.dtype)
    elif len(raw_video.shape) == 3:
        new_video = np.zeros((nt,
                              upscale_factor*nrows,
                              upscale_factor*ncols), dtype=raw_video.dtype)
    else:
        raise RuntimeError("upscale_video_frame does not know how to handle "
                           f"video of shape {raw_video.shape}; must be either "
                           "(nt, nrows, ncols) or (nt, nrow, ncols, ncolors)")

    for ii in range(upscale_factor):
        for jj in range(upscale_factor):
            new_video[:,
                      ii:new_video.shape[1]:upscale_factor,
                      jj:new_video.shape[2]:upscale_factor] = raw_video

    return new_video


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
        (maximum is 9; default is 5)

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

        if quality > 9:
            raise ValueError("You are trying to write an mp4 "
                             f"with quality = {quality}; "
                             "quality > 9 is not compatible with "
                             "all players; please run again with "
                             "different quality setting.")

        self._generator = None  # for bookkeeping; see assign_generator
        self._path = video_path
        self._origin = origin
        self._frame_shape = video_data.shape[1:3]
        if timesteps is None:
            self._timesteps = timesteps
        else:
            self._timesteps = np.copy(timesteps)

        if video_data.shape[1] < 128 or video_data.shape[0] < 128:
            video_data = upscale_video_frame(video_data, 2)

        imageio.mimsave(self._path,
                        video_data,
                        fps=fps,
                        quality=quality,
                        pixelformat='yuv420p',
                        codec='libx264')

    def _assign_generator(self, generator: object) -> None:
        """
        **This should not be used**
        If the ThumbnailVideo is produced by a VideoGenerator,
        the generator uses this method to assign itself to the
        ThumbnailVideo so that the VideoGenerator does not go
        out of scope (thus having its tmp_dir cleaned up) until
        all of the ThumbnailVideos produced from the
        VideoGenerator have gone out of scope.
        """
        if self._generator is not None:
            # should not be able to reaassign self._generator
            return None
        self._generator = generator

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
        timestep_offset: Optional[np.ndarray] = None,
        roi_list: Optional[List[ExtractROI]] = None,
        roi_color: Union[None,
                         Tuple[int, int, int],
                         Dict[int, Tuple[int, int, int]]] = (255, 0, 0)
        ) -> ThumbnailVideo:
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

    roi_list: Optional[List[ExtractROI]]
        If not None, list of ROIs whose contours are to be drawn
        in the thumbnail video (default: None)

    roi_color: Optional[Tuple[int, int, int], Dict[int, Tuple[int, int, int]]]
        RGB color of ROIs to be drawn in the thumbnail video.
        (default (255, 0, 0))
        Or a dict mapping ROI ID to the RGB colors of ROIs

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

    if roi_list is not None:
        # convert to RGB
        if len(sub_video.shape) < 4:
            sub_video = get_rgb_sub_video(sub_video,
                                          (0, 0),
                                          sub_video.shape[1:3])
        sub_video = np.copy(sub_video)

        valid_rois = get_roi_list_in_fov(
                        roi_list,
                        (origin[0]+origin_offset[0],
                         origin[1]+origin_offset[1]),
                        (sub_video.shape[1],
                         sub_video.shape[2]))

        for roi in valid_rois:
            if isinstance(roi_color, dict):
                this_color = roi_color[roi['id']]
            else:
                this_color = roi_color
            sub_video = add_roi_contour_to_video(
                            sub_video,
                            (origin[0]+origin_offset[0],
                             origin[1]+origin_offset[1]),
                            roi,
                            this_color)

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
        min_max: Optional[Tuple[numbers.Number, numbers.Number]] = None,
        quantiles: Optional[Tuple[numbers.Number, numbers.Number]] = None,
        origin_offset: Optional[Tuple[int, int]] = None,
        roi_list: Optional[List[ExtractROI]] = None,
        roi_color: Tuple[int, int, int] = (255, 0, 0)) -> ThumbnailVideo:
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

    min_max: Optional[Tuple[numbers.Number, numbers.Number]]
        If not None, the minimum and maximum values used to clip
        and normalize the movie brightness values (default: None).

    quantiles: Optional[Tuple[numbers.Number, numbers.Number]]
        If not None, the minimum and maximum quantiles used to
        clip and normalize the movie brightness values (default: None)

    origin_offset: Optional[Tuple[int, int]]
        Offset values to be added to origin in container.
        *Should only be used by methods which call this method
        after pre-truncating the video in space; do NOT use this
        by hand*

    roi_list: Optional[List[ExtractROI]]
        If not None, list of ROIs whose contours are to be drawn
        in the thumbnail video (default: None)

    roi_color: Optional[Tuple[int, int, int]]
        RGB color of ROIs to be drawn in the thumbnail video.
        (default (255, 0, 0))

    Returns
    -------
    ThumbnailVideo
        Containing the metadata about the written thumbnail video

    Raises
    ------
    RuntimeError
       If both min_max and quantiles are None or if both
       min_max and quantiles are not None (i.e. one and only
       one of min_max and quantiles must be not None)

    RuntimeError
        If min_max[0] > min_max[1]
    """

    if min_max is None and quantiles is None:
        raise RuntimeError("both min_max and quantiles are None "
                           "in thumbnail_video_from_path; must "
                           "specify one")

    if min_max is not None and quantiles is not None:
        raise RuntimeError("both min_max and quantiles are are not None "
                           "in thumbnail_video_from_path; can only specify "
                           "one")

    data = read_and_scale(full_video_path,
                          origin,
                          frame_shape,
                          quantiles=quantiles,
                          min_max=min_max)

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
                    origin_offset=origin,
                    roi_list=roi_list,
                    roi_color=roi_color)
    return thumbnail


def add_roi_contour_to_video(sub_video: np.ndarray,
                             origin: Tuple[int, int],
                             roi: ExtractROI,
                             roi_color: Tuple[int, int, int]) -> np.ndarray:
    """
    Add the contour of an ROI to a video

    Parameters
    ----------
    sub_video: np.ndarray
        The video as an RGB movie. Shape is (n_t, nrows, ncols, 3)

    origin: Tuple[int, int]
        global (rowmin, colmin) of the video in sub_video

    roi: ExtractROI
        The parameters of this ROI are in global coordinates,
        which is why we need origin as an argument

    roi_color: Tuple[int, int, int]
        RGB color of the ROI contour

    Returns
    -------
    sub_video: np.ndarray
        sub_video with the ROI contour

    Note:
    -----
    While it does return a np.ndarray, this method will change
    sub_video in-place
    """

    n_video_rows = sub_video.shape[1]
    n_video_cols = sub_video.shape[2]

    # construct an ROI object to get the contour mask
    # for us
    ophys_roi = OphysROI(roi_id=-1,
                         x0=roi['x'],
                         y0=roi['y'],
                         width=roi['width'],
                         height=roi['height'],
                         valid_roi=False,
                         mask_matrix=roi['mask'])

    contour_mask = ophys_roi.contour_mask
    for irow in range(contour_mask.shape[0]):
        row = irow+ophys_roi.y0-origin[0]
        if row < 0 or row >= n_video_rows:
            continue
        for icol in range(contour_mask.shape[1]):
            if not contour_mask[irow, icol]:
                continue
            col = icol+ophys_roi.x0-origin[1]
            if col < 0 or col >= n_video_cols:
                continue
            for i_color in range(3):
                sub_video[:, row, col, i_color] = roi_color[i_color]

    return sub_video


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
        padding: int = 0,
        other_roi: Union[None, List[ExtractROI]] = None,
        roi_color: Union[None,
                         Dict[int, Tuple[int, int, int]],
                         Tuple[int, int, int]] = None,
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

    padding: int
        The number of pixels to be added to the FOV beyond
        the ROI bounds (if possible)
        (default = 0)

    other_roi: Union[None, List[ExtractROI]]
        Other ROIs to display

    roi_color: Union[None,
                     Tuple[int, int, int],
                     Dict[int, Tuple[int, int, int]]]
        RGB color in which to draw the ROI in the video;
        or a dict mapping ROI ID to the RGB color
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
        (maximum is 9; default is 5)

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
                                        full_video.shape[1:3],
                                        padding)

    sub_video = get_rgb_sub_video(full_video,
                                  origin,
                                  fov_shape,
                                  timesteps=timesteps)

    # if an ROI color has been specified, plot the ROI
    # contour over the video in the specified color
    roi_list = None
    if roi_color is not None:
        roi_list = [roi]
        if other_roi is not None:
            for roi2 in other_roi:
                if roi2['id'] != roi['id']:
                    roi_list.append(roi2)

    thumbnail = thumbnail_video_from_array(
                    sub_video,
                    (0, 0),
                    sub_video.shape[1:3],
                    timesteps=None,
                    file_path=file_path,
                    tmp_dir=tmp_dir,
                    fps=fps,
                    quality=quality,
                    origin_offset=origin,
                    roi_list=roi_list,
                    roi_color=roi_color)

    return thumbnail


def _thumbnail_video_from_ROI_path(
        video_path: pathlib.Path,
        roi: ExtractROI,
        padding: int = 0,
        other_roi: Union[None, List[ExtractROI]] = None,
        roi_color: Union[None,
                         Tuple[int, int, int],
                         Dict[int, Tuple[int, int, int]]] = None,
        timesteps: Optional[np.ndarray] = None,
        file_path: Optional[pathlib.Path] = None,
        tmp_dir: Optional[pathlib.Path] = None,
        fps: int = 31,
        quality: int = 5,
        min_max: Optional[Tuple[numbers.Number, numbers.Number]] = None,
        quantiles: Optional[Tuple[numbers.Number, numbers.Number]] = None,
        ) -> ThumbnailVideo:
    """
    Get a thumbnail video from a HDF5 file path and an ROI

    Parameters
    ----------
    video_path: pathlib.Path
        path to HDF5 file storing video data.
        Shape of data is (n_times, nrows, ncols)

    roi: ExtractROI

    padding: int
        The number of pixels to be added to the FOV beyond
        the ROI bounds (if possible)
        (default = 0)

    other_roi: Union[None, List[ExtractROI]]
        Other ROIs to display

    roi_color: Union[None,
                     Tuple[int, int, int],
                     Dict[int, Tuple[int, int, int]]]
        RGB color in which to draw the ROI in the video;
        or a dict mapping ROI ID to the RGB color
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
        (maximum is 9; default is 5)

    min_max: Optional[Tuple[numbers.Number, numbers.Number]]
        If not None, the minimum and maximum values used to clip
        and normalize the movie brightness values (default: None).

    quantiles: Optional[Tuple[numbers.Number, numbers.Number]]
        If not None, the minimum and maximum quantiles used to
        clip and normalize the movie brightness values (default: None)

    Returns
    -------
    thumbnail: ThumbnailVideo

    Raises
    ------
    RuntimeError
       If both min_max and quantiles are None or if both
       min_max and quantiles are not None (i.e. one and only
       one of min_max and quantiles must be not None)

    RuntimeError
        If min_max[0] > min_max[1]

    Notes
    -----
    This method will scale video data values to [0, 255]
    """

    if min_max is None and quantiles is None:
        raise RuntimeError("both min_max and quantiles are None "
                           "in thumbnail_video_from_path; must "
                           "specify one")

    if min_max is not None and quantiles is not None:
        raise RuntimeError("both min_max and quantiles are are not None "
                           "in thumbnail_video_from_path; can only specify "
                           "one")

    with h5py.File(video_path, 'r') as in_file:
        img_shape = in_file['data'].shape

    # find bounds of thumbnail
    (origin,
     fov_shape) = video_bounds_from_ROI(roi,
                                        img_shape[1:3],
                                        padding)

    full_video = read_and_scale(video_path,
                                origin,
                                fov_shape,
                                quantiles=quantiles,
                                min_max=min_max)

    sub_video = get_rgb_sub_video(full_video,
                                  (0, 0),
                                  fov_shape,
                                  timesteps=timesteps)

    # if an ROI color has been specified, plot the ROI
    # contour over the video in the specified color
    roi_list = None
    if roi_color is not None:
        roi_list = [roi]
        if other_roi is not None:
            for roi2 in other_roi:
                if roi2['id'] != roi['id']:
                    roi_list.append(roi2)

    thumbnail = thumbnail_video_from_array(
                    sub_video,
                    (0, 0),
                    sub_video.shape[1:3],
                    timesteps=None,
                    file_path=file_path,
                    tmp_dir=tmp_dir,
                    fps=fps,
                    quality=quality,
                    origin_offset=origin,
                    roi_list=roi_list,
                    roi_color=roi_color)

    return thumbnail


def thumbnail_video_from_ROI(
        video: Union[np.ndarray, pathlib.Path],
        roi: ExtractROI,
        padding: int = 0,
        roi_color: Union[None,
                         Tuple[int, int, int],
                         Dict[int, Tuple[int, int, int]]] = None,
        other_roi: Union[None, List[ExtractROI]] = None,
        timesteps: Optional[np.ndarray] = None,
        file_path: Optional[pathlib.Path] = None,
        tmp_dir: Optional[pathlib.Path] = None,
        fps: int = 31,
        quality: int = 5,
        min_max: Optional[Tuple[numbers.Number, numbers.Number]] = None,
        quantiles: Optional[Tuple[numbers.Number, numbers.Number]] = None,
        ) -> ThumbnailVideo:
    """
    Get a thumbnail video from a HDF5 file path and an ROI

    Parameters
    ----------
    video: Union[np.ndarray, pathlib.Path]
        Either a np.ndarray containing video data or the path
        to an HDF5 file containing said array. In either case,
        data is assumed to be shaped like (n_times, nrows, ncols)

    roi: ExtractROI

    padding: int
        The number of pixels to be added to the FOV beyond
        the ROI bounds (if possible)
        (default = 0)

    other_roi: Union[None, List[ExtractROI]]
        Other ROIs to display

    roi_color: Union[None,
                     Tuple[int, int, int],
                     Dict[int, Tuple[int, int, int]]]
        RGB color in which to draw the ROI in the video;
        or a dict mapping ROI ID to the RGB color
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
        (maximum is 9; default is 5)

    min_max: Optional[Tuple[numbers.Number, numbers.Number]]
        If not None, the minimum and maximum values used to clip
        and normalize the movie brightness values (default: None).

    quantiles: Optional[Tuple[numbers.Number, numbers.Number]]
        If not None, the minimum and maximum quantiles used to
        clip and normalize the movie brightness values (default: None)

    Returns
    -------
    thumbnail: ThumbnailVideo

    Notes
    -----
    If video is a np.ndarray, data will not be scaled so that values
    are in [0, 255]. If video is a path, the data will be scaled
    so that values are in [0, 255]

    min_max and quantile are only used if video is a path. In that case
    one and only one of them must be non-None
    """

    if isinstance(video, np.ndarray):
        thumbnail = _thumbnail_video_from_ROI_array(
                           video,
                           roi,
                           padding=padding,
                           other_roi=other_roi,
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
                           padding=padding,
                           other_roi=other_roi,
                           roi_color=roi_color,
                           timesteps=timesteps,
                           file_path=file_path,
                           tmp_dir=tmp_dir,
                           fps=fps,
                           quality=quality,
                           min_max=min_max,
                           quantiles=quantiles)
    else:
        msg = "video must be either a np.ndarray "
        msg += "or a pathlib.Path; you passed in "
        msg += f"{type(video)}"
        raise RuntimeError(msg)
    return thumbnail
