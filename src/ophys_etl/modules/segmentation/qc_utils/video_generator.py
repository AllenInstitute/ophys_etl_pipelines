import h5py
import numpy as np
import tempfile
from pathlib import Path

import ophys_etl.modules.segmentation.qc_utils.video_utils as video_utils


class VideoGenerator(object):
    """
    Class that will actually handle reading data from an HDF5 file
    and writing thumbnail mp4s for viewing in this notebook to a
    temporary directory

    Parameters
    ----------
    video_path
        Path to the HDF5 file containing the full video data
    """
    
    def __init__(self, video_path):
        quantiles = (0.1, 0.999)
        self.tmp_dir = Path(tempfile.mkdtemp(prefix='temp_dir_'))
        with h5py.File(video_path, 'r') as in_file:
            self.min_max = np.quantile(in_file['data'][()], quantiles)
            self.video_shape = in_file['data'].shape

        self.video_path = video_path
                    
    def get_thumbnail_video(self,
                            origin=None,
                            frame_shape=None,
                            timesteps=None,
                            fps=31,
                            quality=5):
        """
        Get a ThumbnailVideo from by-hand specified parameters
        
        Parameters
        ----------
        origin: Tuple[int, int]
            (rowmin, colmin) of the desired thumbnail
        
        frame_shape: Tuple[int, int]
            (nrows, ncols) of the desired thumbprint
        
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
        if origin is None:
            origin = (0, 0)
        if frame_shape is None:
            frame_shape = self.video_shape[1:3]

        thumbnail = video_utils.thumbnail_video_from_path(
                        self.video_path,
                        origin,
                        frame_shape,
                        timesteps=timesteps,
                        tmp_dir=self.tmp_dir,
                        fps=fps,
                        quality=quality,
                        min_max=self.min_max)
        return thumbnail

    def get_thumbnail_video_from_roi(self,
                                     roi=None,
                                     roi_color=None,
                                     timesteps=None,
                                     quality=5,
                                     fps=31):
        """
        Get a ThumbnailVideo from an ROI
        
        Parameters
        ----------
        roi: ExtractROI
        
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
                        roi_color=roi_color,
                        timesteps=timesteps,
                        tmp_dir=self.tmp_dir,
                        fps=fps,
                        quality=quality,
                        min_max=self.min_max)
        return thumbnail
