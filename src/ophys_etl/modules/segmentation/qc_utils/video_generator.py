from typing import Union, Optional, Tuple, List, Dict
import h5py
import numpy as np
import tempfile
import pathlib

from ophys_etl.types import ExtractROI
import ophys_etl.modules.segmentation.qc_utils.video_utils as video_utils


class VideoGenerator(object):
    """
    Class that will actually handle reading data from an HDF5 file
    and writing thumbnail mp4s for viewing in this notebook to a
    temporary directory

    Parameters
    ----------
    video_path: Union[str, pathlib.Path]
        Path to the HDF5 file containing the full video data

    tmp_dir: Optional[pathlib.Path]
        Parent of temporary directory where thumbnail videos
        will be written. If None, tempfile will be used to
        create a temporary directory in /tmp/ (default: None)
    """

    def __init__(self,
                 video_path: Union[str, pathlib.Path],
                 tmp_dir: Optional[pathlib.Path] = None):

        if not isinstance(video_path, pathlib.Path):
            video_path = pathlib.Path(video_path)
        if not video_path.is_file():
            raise RuntimeError(f'{video_path} is not a file')

        # quantiles used to normalize the thumbnail video
        quantiles = (0.1, 0.999)

        if tmp_dir is not None:
            if not tmp_dir.exists():
                tmp_dir.mkdir(parents=True)

        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(dir=tmp_dir,
                                                     prefix='temp_dir_'))

        # read in the video data to learn the shape of the field
        # of view and the minimum/maximum values for normalization
        with h5py.File(video_path, 'r') as in_file:
            self.min_max = np.quantile(in_file['data'][()], quantiles)
            self.video_shape = in_file['data'].shape

        self.video_path = video_path

    def get_thumbnail_video(
            self,
            origin: Optional[Tuple[int, int]] = None,
            frame_shape: Optional[Tuple[int, int]] = None,
            timesteps: Optional[np.ndarray] = None,
            fps: int = 31,
            quality: int = 5,
            rois: Optional[Union[List[ExtractROI],
                           Dict[int, ExtractROI]]] = None,
            roi_color: Tuple[int, int, int] = (255, 0, 0)
            ) -> video_utils.ThumbnailVideo:
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

        Returns
        -------
        video_utils.ThumbnailVideo
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
                roi_list.append(this_roi)

            if len(roi_list) == 0:
                roi_list = None

        thumbnail = video_utils.thumbnail_video_from_path(
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
        return thumbnail

    def get_thumbnail_video_from_roi(
                 self,
                 roi: ExtractROI,
                 padding: int = 0,
                 roi_color: Optional[Tuple[int, int, int]] = None,
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

        roi_color: Optional[Tuple[int, int, int]]
            If not None, the RGB color in which to plot the ROI's
            boundary. If None, ROI is not plotted in thumbnail.
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
        video_utils.ThumbnailVideo
        """
        thumbnail = video_utils.thumbnail_video_from_ROI(
                        self.video_path,
                        roi,
                        padding=padding,
                        roi_color=roi_color,
                        timesteps=timesteps,
                        tmp_dir=self.tmp_dir,
                        fps=fps,
                        quality=quality,
                        min_max=self.min_max)
        return thumbnail
