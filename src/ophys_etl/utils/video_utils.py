import h5py
import tempfile
import multiprocessing
import subprocess
import numpy as np
import imageio_ffmpeg as mpg
from typing import Union, List
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


def transform_to_webm(video: np.ndarray, output_path: str,
                      fps: float, ncpu: int, bitrate: str = "0",
                      crf: int = 20) -> str:
    """Function to transform 2p gray scale video into a webm
    video using imageio_ffmpeg.

    Parameters
    ----------
    video : np.ndarray
        Video to be transformed with shape (time, row, col)
    output_path : str
        Output path for the transformed video
    fps : float
        Desired frames per second (fps) of the output video
    ncpu : int
        Degree of parallelization desired for encoding. Video will be
        split into 'ncpu' parts and each part will be encoded in parallel.
    bitrate : str, optional
        Desired bitrate of output, by default "0". The default *MUST*
        be zero in order to encode in constant quality mode.
    crf : int, optional
        Desired perceptual quality of output, by default 20. Value can
        be from 0 - 63. Lower values mean better quality.

    Returns
    -------
    str
        Output path of the encoded video
    """

    split_video = np.array_split(video, ncpu)
    split_output_paths = [tempfile.NamedTemporaryFile(suffix=f"_{i}.webm")
                          for i in range(ncpu)]

    mp_pool_args = [(vid, outpath.name, fps, bitrate, crf)
                    for vid, outpath in zip(split_video, split_output_paths)]

    with multiprocessing.Pool(ncpu) as pool:
        encode_results = pool.starmap(encode_video, mp_pool_args)

    concat_output = concat_videos(encode_results, output_path)

    for sop in split_output_paths:
        sop.close()

    return concat_output


def concat_videos(video_paths: List[str], output_path: str) -> str:
    """Use ffmpeg to concatenate a list of videos (with the same encoding)
    into a single video.

    Parameters
    ----------
    video_paths : List[str]
        A list of paths (str) for videos that should be concatenated together
        Order of paths matters!
    output_path : str
        The desired output path for the concatenated video

    Returns
    -------
    str
        Path of concatenated video
    """

    with tempfile.NamedTemporaryFile(suffix=".txt") as concat_list_file:
        with open(concat_list_file.name, 'w') as fp:
            for path in video_paths:
                fp.write(f"file '{path}'\n")

        # See: https://trac.ffmpeg.org/wiki/Concatenate
        ffmpeg_concat_cmd = [mpg.get_ffmpeg_exe(), '-y', '-f', 'concat',
                             '-safe', '0', '-i', concat_list_file.name,
                             '-c', 'copy', output_path]
        subprocess.run(ffmpeg_concat_cmd)

    return output_path


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
    pix_fmt_out : str
        Desired pixel format for output encoded video, by default "yuv420p"
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
