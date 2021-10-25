from typing import Tuple, Dict, List, Union
import h5py
import numpy as np
import pathlib

from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysROI)

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
    mn, mx = np.quantile(img_data, quantiles)
    out_img = np.where(img_data > mn, img_data, mn)
    out_img = np.where(out_img < mx, out_img, mx)
    return out_img


def scale_img_to_uint8(
        img_data: np.ndarray) -> np.ndarray:
    """
    Scale an image to np.uint8
    """
    min_val = img_data.min()
    max_val = img_data.max()
    delta = max_val-min_val
    out_img = np.round(255.0*(img_data.astype(float)-min_val)/delta)
    return out_img.astype(np.uint8)
