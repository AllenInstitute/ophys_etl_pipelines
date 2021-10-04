from typing import Tuple, Union, List, Optional
import pathlib
import numpy as np

import logging
import time

from ophys_etl.utils.array_utils import (
    pairwise_distances)

from ophys_etl.modules.segmentation.merge.candidates import (
    create_neighbor_lookup)

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.utils.roi_utils import (
    pixel_list_to_extract_roi,
    extract_roi_to_ophys_roi)

from ophys_etl.modules.segmentation.merge.louvain_utils import (
    correlate_all_pixels,
    _do_louvain_clustering)


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def do_louvain_clustering_on_rois(
        roi_list: List[OphysROI],
        full_video: np.ndarray,
        kernel_size: Union[float, None],
        filter_fraction: float,
        n_processors: int,
        scratch_dir: pathlib.Path,
        correlation_floor: float = 0.0,
        only_neighbors: Optional[bool] = False) -> Tuple[List[OphysROI],
                                                         List[List[int]]]:
    """
    Run Louvain-based merger algorithm on a list of ROIs.

    Parameters
    -----------
    roi_list: List[OphysROI]

    full_video: np.ndarray
        (n_time, n_rows, n_cols) video of the full field of view

    kernel_size: Union[float, None]
        If not None, only pixels this far away from each other
        or closer will be allowed to have non-zero correlation

    filter_fraction: float
        The fraction of brightest timesteps to use when correlating
        two pixels

    n_processors: int
        The number of multiprocessing.Processes to use during
        parallel computations

    scratch_dir: pathlib.Path
        Path to a directory where an intermediate HDF5 file may be
        written out

    correlation_floor: float
        Correlation values beneath this value will be set to zero
        (should always be set to at least zero; negative correlation
        values cause unexpected behavior in the algorithm)

    only_neighbors: bool
        If True, only ever consider mergers between two ROIs that
        are contiguous.

    Return
    ------
    merged_roi_list: List[OphysROI]

    merger_history: List[List[int]]
        A list of mergers of the form ['absorber', 'absorbed']
    """
    if not only_neighbors:
        neighbor_lookup = None
    else:
        neighbor_lookup = create_neighbor_lookup(
                                {roi.roi_id: roi
                                 for roi in roi_list},
                                n_processors)

    n_time = full_video.shape[0]
    all_pixel_set = set()
    for roi in roi_list:
        all_pixel_set = all_pixel_set.union(roi.global_pixel_set)
    n_pixels = len(all_pixel_set)
    sub_video = np.zeros((n_time, n_pixels), dtype=float)
    index_to_pixel = []
    roi_id_array = []

    # don't really know how to handle overlaps here
    already_added_pixels = set()
    i_pixel = 0
    for roi in roi_list:
        for pixel in roi.global_pixel_array:
            pixel = tuple(pixel)
            if pixel in already_added_pixels:
                continue
            trace = full_video[:, pixel[0], pixel[1]]
            sub_video[:, i_pixel] = trace
            index_to_pixel.append(pixel)
            roi_id_array.append(roi.roi_id)
            i_pixel += 1
            already_added_pixels.add(pixel)

    roi_id_array = np.array(roi_id_array)

    if kernel_size is None:
        pixel_distances = None
    else:
        pixel_coord_array = np.array([[p[0], p[1]]
                                      for p in index_to_pixel])
        pixel_distances = pairwise_distances(pixel_coord_array)

    t0 = time.time()
    correlation_matrix = correlate_all_pixels(
                            sub_video,
                            filter_fraction,
                            n_processors,
                            scratch_dir,
                            pixel_distances=pixel_distances,
                            kernel_size=kernel_size)

    duration = time.time()-t0
    logger.info(f'calculated correlation matrix in {duration:.2f} seconds')

    (merged_roi_id_array,
     merger_history) = _do_louvain_clustering(
                              roi_id_array,
                              correlation_matrix,
                              neighbor_lookup=neighbor_lookup,
                              n_processors=n_processors,
                              correlation_floor=correlation_floor)

    new_roi_list = []
    for roi_id in np.unique(merged_roi_id_array):
        pixel_list = [index_to_pixel[ii]
                      for ii in np.where(merged_roi_id_array == roi_id)[0]]
        new_extract_roi = pixel_list_to_extract_roi(pixel_list, roi_id)
        new_ophys_roi = extract_roi_to_ophys_roi(new_extract_roi)
        new_roi_list.append(new_ophys_roi)

    output_merger_history = [[merger_history[roi_id], roi_id]
                             for roi_id in merger_history]

    return (new_roi_list,
            output_merger_history)
