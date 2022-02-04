from typing import Tuple, Optional, Union
import h5py
import numpy as np
import imageio_ffmpeg as mpg
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


def scale_video_to_uint8(video: np.ndarray,
                         min_value: Union[int, float],
                         max_value: Union[int, float]) -> np.ndarray:
    """
    Convert a video (as a numpy.ndarray) to uint8 by dividing by the
    array's maximum value and multiplying by 255

    Parameters
    ----------
    video: np.ndarray

    min_value: Optional[Union[int, float]]

    max_value: Optional[Union[int, float]]
        Video will be clipped at min_value and max_value and
        then normalized to (max_value-min_value) before being
        converted to uint8

    Returns
    -------
    np.ndarray

    Raises
    ------
    RuntimeError
        If min_value > max_value
    """

    if min_value > max_value:
        raise RuntimeError("in scale_video_to_uint8 "
                           f"min_value ({min_value}) > "
                           f"max_value ({max_value})")

    mask = video > max_value
    video[mask] = max_value
    mask = video < min_value
    video[mask] = min_value

    delta = (max_value-min_value)
    video = video-min_value
    return np.round(255*video.astype(float)/delta).astype(np.uint8)


def _read_and_scale_all_at_once(
        full_video_path: Path,
        origin: Tuple[int, int],
        frame_shape: Tuple[int, int],
        quantiles: Optional[Tuple[float, float]] = None,
        min_max: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Read in a video from an HDF5 file and scale it to np.uint8
    without chunking

    Parameters
    ----------
    full_video_path: pathlib.Path
        Path to the HDF5 file

    origin: Tuple[int, int]
        Origin of the desired field of view

    frame_shape: Tuple[int, int]
        Shape of the desired field of view

    quantiles: Optional[Tuple[float, float]]
        Quantiles of full video used for scale normalization
        (default: None)

    min_max: Optional[Tuple[float, float][
        Minimum and maximum values used for scale normalization
        (default: None)

    Returns
    -------
    data: np.ndarray
        Video, cropped to the specified field of view and scaled
        to np.uint8 (i.e. dynamic range is [0, 255])

    Notes
    -----
    One and only one of quantiles, min_max must be specified. If
    both or neither are specified, a RuntimeError will be raised.
    """

    if quantiles is None and min_max is None:
        raise RuntimeError("must specify either quantiles or min_max "
                           "in _read_and_scale_all_at_once; both are None")
    if quantiles is not None and min_max is not None:
        raise RuntimeError("cannot specify both quantiles and min_max "
                           "in _read_and_scale_all_at_once")

    with h5py.File(full_video_path, 'r') as in_file:
        if quantiles is not None:
            read_data = in_file['data'][()]
            min_max = np.quantile(read_data, quantiles)
            data = read_data[:,
                             origin[0]:origin[0]+frame_shape[0],
                             origin[1]:origin[1]+frame_shape[1]]
            del read_data
        else:
            data = in_file['data'][:,
                                   origin[0]:origin[0]+frame_shape[0],
                                   origin[1]:origin[1]+frame_shape[1]]

    if min_max[0] > min_max[1]:
        raise RuntimeError(f"min_max {min_max} in "
                           "_read_and_scale_all_at_once; "
                           "order seems to be reversed")

    data = scale_video_to_uint8(data, min_max[0], min_max[1])
    return data


def _read_and_scale_by_chunks(
        full_video_path: Path,
        origin: Tuple[int, int],
        frame_shape: Tuple[int, int],
        quantiles: Optional[Tuple[int, int]] = None,
        min_max: Optional[Tuple[int, int]] = None,
        time_chunk_size: int = 100) -> np.ndarray:

    """
    Read in a video from an HDF5 file and scale it to np.uint8
    one chunk at a time

    Parameters
    ----------
    full_video_path: pathlib.Path
        Path to the HDF5 file

    origin: Tuple[int, int]
        Origin of the desired field of view

    frame_shape: Tuple[int, int]
        Shape of the desired field of view

    quantiles: Optional[Tuple[float, float]]
        Quantiles of full video used for scale normalization
        (default: None)

    min_max: Optional[Tuple[float, float][
        Minimum and maximum values used for scale normalization
        (default: None)

    time_chunk_size: int
        Number of time steps to process at once.
        (default: 100)

    Returns
    -------
    data: np.ndarray
        Video, cropped to the specified field of view and scaled
        to np.uint8 (i.e. dynamic range is [0, 255])

    Notes
    -----
    One and only one of quantiles, min_max must be specified. If
    both or neither are specified, a RuntimeError will be raised.
    """

    if quantiles is None and min_max is None:
        raise RuntimeError("must specify either quantiles or min_max "
                           "in _read_and_scale_by_chunk; both are None")
    if quantiles is not None and min_max is not None:
        raise RuntimeError("cannot specify both quantiles and min_max "
                           "in _read_and_scale_by_chunks")

    with h5py.File(full_video_path, 'r') as in_file:
        dataset = in_file['data']
        rowmin = origin[0]
        rowmax = min(dataset.shape[1], origin[0]+frame_shape[0])
        colmin = origin[1]
        colmax = min(dataset.shape[2], origin[1]+frame_shape[1])

        if quantiles is not None:
            min_max = np.quantile(dataset[()], quantiles)

        if min_max[0] > min_max[1]:
            raise RuntimeError(f"min_max {min_max} in "
                               "_read_and_scale_by_chunks; "
                               "order seems to be reversed")

        nt = dataset.shape[0]
        final_output = np.zeros((nt, rowmax-rowmin, colmax-colmin),
                                dtype=np.uint8)

        for t0 in range(0, nt, time_chunk_size):
            t1 = min(t0+time_chunk_size, nt)
            data_chunk = scale_video_to_uint8(dataset[t0:t1,
                                                      rowmin:rowmax,
                                                      colmin:colmax],
                                              min_max[0],
                                              min_max[1])

            final_output[t0:t1, :, :] = data_chunk

    return final_output


def read_and_scale(
        video_path: Path,
        origin: Tuple[int, int],
        frame_shape: Tuple[int, int],
        quantiles: Optional[Tuple[float, float]] = None,
        min_max: Optional[Tuple[float, float]] = None) -> np.ndarray:
    """
    Read in a video from an HDF5 file and scale it to np.uint8

    Parameters
    ----------
    video_path: pathlib.Path
        Path to the HDF5 file

    origin: Tuple[int, int]
        Origin of the desired field of view

    frame_shape: Tuple[int, int]
        Shape of the desired field of view

    quantiles: Optional[Tuple[float, float]]
        Quantiles of full video used for scale normalization
        (default: None)

    min_max: Optional[Tuple[float, float][
        Minimum and maximum values used for scale normalization
        (default: None)

    Returns
    -------
    data: np.ndarray
        Video, cropped to the specified field of view and scaled
        to np.uint8 (i.e. dynamic range is [0, 255])

    Notes
    -----
    One and only one of quantiles, min_max must be specified. If
    both or neither are specified, a RuntimeError will be raised.

    If the area of the requested field of view is < 2500, the
    movie will be read in and scaled all at once. Otherwise, it
    will be scaled one chunk at a time.
    """

    if quantiles is None and min_max is None:
        raise RuntimeError("must specify either quantiles or min_max "
                           "in read_and_scale; both are None")
    if quantiles is not None and min_max is not None:
        raise RuntimeError("cannot specify both quantiles and min_max "
                           "in read_and_scale")

    area = frame_shape[0]*frame_shape[1]
    if area < 2500:
        return _read_and_scale_all_at_once(
                       video_path,
                       origin,
                       frame_shape,
                       quantiles=quantiles,
                       min_max=min_max)

    return _read_and_scale_by_chunks(
                   video_path,
                   origin,
                   frame_shape,
                   quantiles=quantiles,
                   min_max=min_max)
