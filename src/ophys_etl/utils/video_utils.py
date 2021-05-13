import h5py
import numpy as np
import imageio_ffmpeg as mpg
from typing import Union
from pathlib import Path
from ophys_etl.utils.array_utils import downsample_array


def downsample_h5_video(
        video_path: Union[Path],
        input_fps: float = 31.0,
        output_fps: float = 4.0,
        strategy: str = 'average',
        random_seed: int = 0) -> np.ndarray:
    """Opens an h5 file and downsamples dataset 'data'
    along axis=0

    Parameters
    ----------
        video_path: pathlib.Path
            path to an h5 video. Should have dataset 'data'. For video,
            assumes dimensions [time, width, height] and downsampling
            applies to time.
        input_fps: float
            frames-per-second of the input array
        output_fps: float
            frames-per-second of the output array
        strategy: str
            downsampling strategy. 'random', 'maximum', 'average',
            'first', 'last'. Note 'maximum' is not defined for
            multi-dimensional arrays
        random_seed: int
            passed to numpy.random.default_rng if strategy is 'random'

    Returns:
        video_out: numpy.ndarray
            array downsampled along axis=0
    """
    with h5py.File(video_path, 'r') as h5f:
        video_out = downsample_array(
                h5f['data'],
                input_fps,
                output_fps,
                strategy,
                random_seed)
    return video_out


def encode_video(video: np.ndarray, output_path: str,
                 fps: float, bitrate: str = "0", crf: int = 20) -> str:
    """Encode a video with vp9 codec via imageio-ffmpeg

    Parameters
    ----------
    video : np.ndarray
        Video to be encoded
    output_path : str
        Desired output path for encoded video
    fps : float
        Desired frame rate for encoded video
    bitrate : str, optional
        Desired bitrate of output, by default "0". The default *MUST*
        be zero in order to encode in constant quality mode. Other values
        will result in constrained quality mode.
    crf : int, optional
        Desired perceptual quality of output, by default 20. Value can
        be from 0 - 63. Lower values mean better quality (but bigger video
        sizes).

    Returns
    -------
    str
        Output path of the encoded video
    """

    # ffmpeg expects video shape in terms of: (width, height)
    video_shape = (video[0].shape[1], video[0].shape[0])

    writer = mpg.write_frames(output_path,
                              video_shape,
                              pix_fmt_in="gray8",
                              pix_fmt_out="yuv420p",
                              codec="libvpx-vp9",
                              fps=fps,
                              bitrate=bitrate,
                              output_params=["-crf", str(crf)])

    writer.send(None)  # Seed ffmpeg-imageio writer generator
    for frame in video:
        writer.send(frame)
    writer.close()

    return output_path
