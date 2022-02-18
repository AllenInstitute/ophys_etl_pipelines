from typing import Tuple, Dict, List, Union, Optional
import h5py
import numpy as np
import pathlib
import PIL.Image
import copy

from ophys_etl.types import (
    OphysROI)

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.utils.array_utils import normalize_array

import time
import logging

import hashlib


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def file_hash_from_path(file_path: Union[str, pathlib.Path]) -> str:
    """
    Return the hexadecimal file hash for a file

    Parameters
    ----------
    file_path: Union[str, Path]
        path to a file

    Returns
    -------
    str:
        The file hash (md5; hexadecimal) of the file
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as in_file:
        chunk = in_file.read(1000000)
        while len(chunk) > 0:
            hasher.update(chunk)
            chunk = in_file.read(1000000)
    return hasher.hexdigest()


def get_traces(
        video_path: pathlib.Path,
        ophys_roi_list: List[OphysROI]) -> Dict[int, np.ndarray]:
    """
    Read in the path to a video and a list of OPhysROI.
    Return a dict mapping ROI ID to the trace of the ROI.

    Parameters
    ----------
    video_path: pathlib.Path

    ophys_roi_list: List[OphysROI]

    Returns
    -------
    Dict[int, np.ndarray]
    """
    with h5py.File(video_path, 'r') as in_file:
        video_data = in_file['data'][()]

    t0 = time.time()
    ct = 0
    trace_lookup = {}
    for roi in ophys_roi_list:
        rows = roi.global_pixel_array[:, 0]
        cols = roi.global_pixel_array[:, 1]
        trace = video_data[:, rows, cols]
        trace = np.mean(trace, axis=1)
        trace_lookup[roi.roi_id] = trace
        ct += 1
        if ct % 100 == 0:
            d = time.time()-t0
            p = d/ct
            pred = p*len(ophys_roi_list)
            r = pred-d
            logger.info(f'{ct} traces in {d:.2e} seconds'
                        f' -- {r:.2e} seconds remain of '
                        f'estimated {pred:.2e}')
    return trace_lookup


def clip_img_to_quantiles(
        img_data: np.ndarray,
        quantiles: Tuple[float, float]) -> np.ndarray:
    """
    Clip an image at specified quantiles.
    """
    (min_quantile,
     max_quantile) = np.quantile(img_data, quantiles)
    out_img = np.clip(img_data, min_quantile, max_quantile)
    return out_img


def create_metadata_entry(
        file_path: pathlib.Path) -> Dict[str, str]:
    """
    Create the metadata entry for a file path

    Parameters
    ----------
    file_path: pathlib.Path
        Path to the file whose metadata you want

    Returns
    -------
    metadata: Dict[str, str]
        'path' : absolute path to the file
        'hash' : hexadecimal hash of the file
    """
    hash_value = file_hash_from_path(file_path)
    return {'path': str(file_path.resolve().absolute()),
            'hash': hash_value}


def create_metadata(input_args: dict,
                    video_path: pathlib.Path,
                    roi_path: pathlib.Path,
                    correlation_path: pathlib.Path,
                    motion_csv_path: Optional[pathlib.Path] = None) -> dict:
    """
    Create the metadata dict for an artifact file

    Parameters
    ----------
    input_args: dict
        The arguments passed to the ArtifactGenerator

    video_path: pathlib.Path
        path to the video file

    roi_path: pathlib.Path
        path to the serialized ROIs

    correlation_path: pathlib.Path
        path to the correlation projection data

    motion_csv_path: Optional[pathlib.Path]
        path to the csv file from which the motion border is read

    Returns
    -------
    metadata: dict
        The complete metadata for the artifact file
    """
    metadata = dict()
    metadata['generator_args'] = copy.deepcopy(input_args)

    metadata['video'] = create_metadata_entry(video_path)
    metadata['rois'] = create_metadata_entry(roi_path)
    metadata['correlation'] = create_metadata_entry(correlation_path)
    if motion_csv_path is not None:
        metadata['motion_csv'] = create_metadata_entry(motion_csv_path)

    return metadata


def create_max_and_avg_projections(
        video_path: pathlib.Path,
        lower_quantile: float,
        upper_quantile: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute maximum and average projection images for a video

    Parameters
    ----------
    video_path: pathlib.Path
        path to video file

    lower_quantile: float
        lower quantile to clip the projections to

    upper_quantile: float
        upper quantile to clip the projections to

    Returns
    -------
    average_projection: np.ndarray

    max_projection: np.ndarray
        Both arrays of np.uint8
    """
    with h5py.File(video_path, 'r') as in_file:
        raw_video_data = in_file['data'][()]
    max_img_data = np.max(raw_video_data, axis=0)
    avg_img_data = np.mean(raw_video_data, axis=0)

    max_img_data = clip_img_to_quantiles(
                       max_img_data,
                       (lower_quantile,
                        upper_quantile))

    avg_img_data = clip_img_to_quantiles(
                       avg_img_data,
                       (lower_quantile,
                        upper_quantile))

    return avg_img_data, max_img_data


def create_correlation_projection(
        file_path: pathlib.Path) -> np.ndarray:
    """
    Parameters
    ----------
    file_path: pathlib.Path
        Path to correlation projection data (either pkl data
        containing a graph or png file containing an image)

    Returns
    -------
    correlation_projection: np.ndarray
        Scaled to np.uint8
    """
    if str(file_path).endswith('png'):
        correlation_img_data = np.array(
                                   PIL.Image.open(
                                       file_path, 'r'))
    else:
        correlation_img_data = graph_to_img(file_path)

    correlation_img_data = normalize_array(correlation_img_data)
    return correlation_img_data
