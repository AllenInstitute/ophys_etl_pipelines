from typing import Union, Optional, Tuple, List, Dict
import h5py
import numpy as np
import tempfile
import pathlib

from ophys_etl.types import ExtractROI
import ophys_etl.utils.thumbnail_video_utils as thumbnail_utils


class VideoGenerator(object):
    """
    Class that will actually handle reading data from an HDF5 file
    and writing thumbnail mp4s for viewing in this notebook to a
    temporary directory

    Parameters
    ----------
    video_path: Union[str, pathlib.Path, None]
        Path to the HDF5 file containing the full video data

    video_data: Union[np.ndarray, None]
        A (ntime, nrows, ncols) array to directly use as video
        data.

    quantiles: Tuple[float, float]
        Quantiles used to clip video. Only used if video is specified
        via video_path. Ignored if video is specified via video_data.

    tmp_dir: Optional[pathlib.Path]
        Parent of temporary directory where thumbnail videos
        will be written. If None, tempfile will be used to
        create a temporary directory in /tmp/ (default: None)

    Raises
    ------
    RuntimeError if both or neither video_path and video_data are
    not None (i.e. you must specify one and only one of them).
    """

    def __init__(self,
                 video_path: Union[str, pathlib.Path, None] = None,
                 video_data: Union[np.ndarray, None] = None,
                 quantiles: Tuple[float, float] = (0.1, 0.999),
                 tmp_dir: Optional[pathlib.Path] = None):

        if video_path is None and video_data is None:
            raise RuntimeError("must specify either video_path or video_data")

        if video_path is not None and video_data is not None:
            raise RuntimeError("cannot specify both video_path and video_data")

        if video_path is not None:
            if not isinstance(video_path, pathlib.Path):
                video_path = pathlib.Path(video_path)
            if not video_path.is_file():
                raise RuntimeError(f'{video_path} is not a file')

        if tmp_dir is not None:
            if not tmp_dir.exists():
                tmp_dir.mkdir(parents=True)

        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(dir=tmp_dir,
                                                     prefix='temp_dir_'))

        self._video_data = None
        self._video_path = None
        self._video_shape = None
        self._min_max = None
        self._use_video_data = False

        if video_path is not None:
            # read in the video data to learn the shape of the field
            # of view and the minimum/maximum values for normalization
            with h5py.File(video_path, 'r') as in_file:
                self._min_max = np.quantile(in_file['data'][()], quantiles)
                self._video_shape = in_file['data'].shape

            self._video_path = video_path
        else:
            self._use_video_data = True
            self._video_data = video_data
            self._video_shape = self._video_data.shape

    def __del__(self):
        if hasattr(self, 'tmp_dir'):
            if self.tmp_dir.is_dir():
                self.tmp_dir.rmdir()

    @property
    def video_path(self):
        if self._video_path is None:
            raise RuntimeError("cannot access video_path; it is None")
        return self._video_path

    @property
    def video_data(self):
        if self._video_data is None:
            raise RuntimeError("cannot access video_data; it is None")
        return self._video_data

    @property
    def min_max(self):
        if self._min_max is None:
            raise RuntimeError("cannot access min_max; it is None")
        return self._min_max

    @property
    def video_shape(self):
        return self._video_shape

    def get_thumbnail_video(
            self,
            origin: Optional[Tuple[int, int]] = None,
            frame_shape: Optional[Tuple[int, int]] = None,
            timesteps: Optional[np.ndarray] = None,
            fps: int = 31,
            quality: int = 5,
            rois: Optional[Union[List[ExtractROI],
                           Dict[int, ExtractROI]]] = None,
            roi_color: Tuple[int, int, int] = (255, 0, 0),
            valid_only: bool = False
            ) -> thumbnail_utils.ThumbnailVideo:
        """
        Get a ThumbnailVideo from by-hand specified parameters

        Parameters
        ----------
        origin: Optional[Tuple[int, int]]
            (rowmin, colmin) of the desired thumbnail.
            If None, set to (0, 0) (default=None)

        frame_shape: Tuple[int, int]
            (nrows, ncols) of the desired thumbprint.
            If None, use the whole field of view (default=None)

        timesteps: Optional[np.ndarray]
            If not None, timesteps to put in the thumbnail
            video. If None, use all timesetps (default: None)

        fps: int
            frames per second (default: 31)

        quality: int
            quality parameter passed to imageio. Max is 10.
            (default: 5)

        rois: Optional[Union[List[ExtractROI], Dict[int, ExtractROI]]]
            ROIs to overplot on video. Either a list or a dict that
            maps roi_id to ROI (default: None)

        roi_color: Tuple[int, int, int]
            RGB color to plot ROIs (default (255, 0, 0))

        valid_only: bool
            If rois is not None, only plot valid ROIs
            (default: False)

        Returns
        -------
        thumbnail_utils.ThumbnailVideo
        """
        if origin is None:
            origin = (0, 0)
        if frame_shape is None:
            frame_shape = self.video_shape[1:3]

        roi_list = None
        if rois is not None:
            # select only ROIs that have a hope of intersecting
            # with the field of view

            if isinstance(rois, list):
                raw_roi_list = rois
            else:
                raw_roi_list = list(rois.values())

            rowmin = origin[0]
            rowmax = origin[0] + frame_shape[0]
            colmin = origin[1]
            colmax = origin[1] + frame_shape[1]
            roi_list = []
            for this_roi in raw_roi_list:
                roi_r0 = this_roi['y']
                roi_r1 = this_roi['y'] + this_roi['height']
                roi_c0 = this_roi['x']
                roi_c1 = this_roi['x'] + this_roi['width']
                if roi_r1 < rowmin:
                    continue
                if roi_r0 > rowmax:
                    continue
                if roi_c1 < colmin:
                    continue
                if roi_c0 > colmax:
                    continue
                if valid_only and not this_roi['valid']:
                    continue
                roi_list.append(this_roi)

            if len(roi_list) == 0:
                roi_list = None

        if self._use_video_data:
            thumbnail = thumbnail_utils.thumbnail_video_from_array(
                            self.video_data,
                            origin,
                            frame_shape,
                            timesteps=timesteps,
                            tmp_dir=self.tmp_dir,
                            fps=fps,
                            quality=quality,
                            roi_list=roi_list,
                            roi_color=roi_color)
        else:
            thumbnail = thumbnail_utils.thumbnail_video_from_path(
                            self.video_path,
                            origin,
                            frame_shape,
                            timesteps=timesteps,
                            tmp_dir=self.tmp_dir,
                            fps=fps,
                            quality=quality,
                            min_max=self.min_max,
                            roi_list=roi_list,
                            roi_color=roi_color)

        # so this generator cannot go out of scope before
        # the ThumbnailVideos it produces go out of scope
        thumbnail._assign_generator(self)

        return thumbnail

    def get_thumbnail_video_from_roi(
                 self,
                 roi: ExtractROI,
                 padding: int = 0,
                 other_roi: Union[None, List[ExtractROI]] = None,
                 roi_color: Union[None,
                                  Tuple[int, int, int],
                                  Dict[int, Tuple[int, int, int]]] = None,
                 timesteps: Optional[np.ndarray] = None,
                 quality: int = 5,
                 fps: int = 31):
        """
        Get a ThumbnailVideo from an ROI

        Parameters
        ----------
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

        Returns
        -------
        thumbnail_utils.ThumbnailVideo
        """
        if self._use_video_data:
            video_arg = self.video_data
            min_max = None
        else:
            video_arg = self.video_path
            min_max = self.min_max

        thumbnail = thumbnail_utils.thumbnail_video_from_ROI(
                        video_arg,
                        roi,
                        padding=padding,
                        other_roi=other_roi,
                        roi_color=roi_color,
                        timesteps=timesteps,
                        tmp_dir=self.tmp_dir,
                        fps=fps,
                        quality=quality,
                        min_max=min_max)

        # so this generator cannot go out of scope before
        # the ThumbnailVideos it produces go out of scope
        thumbnail._assign_generator(self)

        return thumbnail
