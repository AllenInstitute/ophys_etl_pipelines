import json
import logging
import h5py
import numpy as np
from typing import List, Union, Dict, Tuple
from pathlib import Path

from ophys_etl.utils.roi_masks import (
    NeuropilMask,
    create_roi_mask_array,
    validate_mask,
    create_roi_masks,
    Mask,
)


def calculate_traces(
    stack: np.ndarray, mask_list: List[Mask], block_size: int = 1000
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Calculates the average response of the specified masks in the
    image stack

    Parameters
    ----------
    stack: float[image height][image width]
        Image stack that masks are applied to

    mask_list: list<Mask>
        List of masks

    Returns
    -------
    traces: float[number masks][number frames]
        This is the average response for each Mask in each image frame
    exclusions: list
        each item like {"roi_id": 123, "exclusion_label_name": "name"}

    """

    traces = np.zeros((len(mask_list), stack.shape[0]), dtype=float)
    num_frames = stack.shape[0]

    mask_areas = np.zeros(len(mask_list), dtype=float)
    valid_masks = np.ones(len(mask_list), dtype=bool)

    exclusions = []

    for i, mask in enumerate(mask_list):
        current_exclusions = validate_mask(mask)
        if len(current_exclusions) > 0:
            traces[i, :] = np.nan
            valid_masks[i] = False
            exclusions.extend(current_exclusions)
            reasons = ", ".join(
                [item["exclusion_label_name"] for item in current_exclusions]
            )
            logging.warning(
                "unable to extract traces for mask "
                '"{}": {} '.format(mask.label, reasons)
            )
            continue

        if not isinstance(mask.mask, np.ndarray):
            mask.mask = np.array(mask.mask)
        mask_areas[i] = mask.mask.sum()

    # calculate traces
    for frame_num in range(0, num_frames, block_size):
        if frame_num % block_size == 0:
            logging.debug("frame " + str(frame_num) + " of " + str(num_frames))
        frames = stack[frame_num : frame_num + block_size]

        for i in range(len(mask_list)):
            if not valid_masks[i]:
                continue

            mask = mask_list[i]
            subframe = frames[
                :, mask.y : mask.y + mask.height, mask.x : mask.x + mask.width
            ]

            total = subframe[:, mask.mask].sum(axis=1)
            traces[i, frame_num : frame_num + block_size] = (
                total / mask_areas[i]
            )

    return traces, exclusions


def calculate_roi_and_neuropil_traces(
    movie_h5: Union[str, Path, np.ndarray],
    roi_mask_list: List[Mask],
    motion_border: List[int],
) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
    """
    calculates neuropil masks for each ROI and then calculates both ROI trace
    and neuropil trace

    Parameters
    ----------
    movie_h5: str, Path, np.ndarray
        location on disk of the movie, or the data array
    roi_mask_list: list<Mask>
        List of masks
    motion_border: list(int)
        the 4 motion border values

    Returns
    -------
    roi_traces: float[number masks][number frames]
        This is the average response for each Mask in each image frame
    neuropil_traces: float[number masks][number frames]
        This is the average response for each Mask in each image frame
    exclusions: list
        each item like {"roi_id": 123, "exclusion_label_name": "name"}

    """

    # a combined binary mask for all ROIs (this is used to
    #   subtracted ROIs from annuli
    mask_array = create_roi_mask_array(roi_mask_list)
    combined_mask = mask_array.max(axis=0)

    logging.info("%d total ROIs" % len(roi_mask_list))

    # create neuropil masks for the central ROIs
    neuropil_mask_list = []
    for m in roi_mask_list:
        nmask = NeuropilMask.create_neuropil_mask(
            m, motion_border, combined_mask, "neuropil for " + m.label
        )
        neuropil_mask_list.append(nmask)

    num_rois = len(roi_mask_list)
    # read the large image stack only once
    combined_list = roi_mask_list + neuropil_mask_list

    if isinstance(movie_h5, np.ndarray):
        # decrosstalk/ophys_plane was using a version of this function
        # that already had the data loaded.
        traces, exclusions = calculate_traces(movie_h5, combined_list)
    else:
        # trace extraction was using a version loading from file
        with h5py.File(movie_h5, "r") as movie_f:
            stack_frames = movie_f["data"]
            traces, exclusions = calculate_traces(stack_frames, combined_list)

    roi_traces = traces[:num_rois]
    neuropil_traces = traces[num_rois:]

    return roi_traces, neuropil_traces, neuropil_mask_list, exclusions


def write_trace_file(data: np.ndarray, names: List[str], path: str):
    logging.debug("Writing {}".format(path))

    utf_dtype = h5py.special_dtype(vlen=str)
    with h5py.File(path, "w") as fil:
        fil["data"] = data
        fil.create_dataset(
            "roi_names",
            data=np.array(names).astype(np.string_),
            dtype=utf_dtype,
        )


def create_roi_dict(id, x, y, width, height, mask):
    return {
        "id": id,
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
        "mask": mask.astype(int).tolist(),
    }


def write_mask_file(
    neuropil_mask_list: List[NeuropilMask], ids: List[str], path: str
):
    logging.debug("Writing {}".format(path))
    masks = [mask.mask for mask in neuropil_mask_list]
    x_coordinates = [mask.x for mask in neuropil_mask_list]
    y_coordinates = [mask.y for mask in neuropil_mask_list]
    widths = [mask.width for mask in neuropil_mask_list]
    heights = [mask.height for mask in neuropil_mask_list]
    output = {
        "neuropils": [
            create_roi_dict(id, x, y, width, height, mask)
            for id, x, y, width, height, mask in zip(
                ids, x_coordinates, y_coordinates, widths, heights, masks
            )
        ]
    }
    with open(path, "w") as f:
        json.dump(output, f)


def extract_traces(
    motion_corrected_stack: Union[str, Path],
    motion_border: Dict,
    storage_directory: Union[str, Path],
    rois: List[Dict],
) -> Dict:
    """
    calculates neuropil masks for ROIs (LIMS format),
    calculates both ROI trace and neuropil trace,
    writes traces to files

    Parameters
    ----------
    motion_corrected_stack: str, Path
        location on disk of the movie
    motion_border: dict
        the motion border dictionary
        ophys_etl.modules.trace_extraction.schemas.MotionBorder
    storage_directory: str, Path
        the output directory
    rois: dict
        List of ROIs in LIMS format

    Returns
    -------
    Dict
        ophys_etl.modules.trace_extraction.schemas.TraceExtractionOutputSchema
        keys:
        - neuropil_trace_file
        - roi_trace_file
        - exclusion_labels

    """

    # find width and height of movie
    with h5py.File(motion_corrected_stack, "r") as f:
        height, width = f["data"].shape[1:]

    border = [
        motion_border["x0"],
        motion_border["x1"],
        motion_border["y0"],
        motion_border["y1"],
    ]

    # create roi mask objects
    roi_mask_list = create_roi_masks(rois, width, height, border)
    roi_names = [roi.label for roi in roi_mask_list]

    # extract traces
    (
        roi_traces,
        neuropil_traces,
        neuropil_mask_list,
        exclusions,
    ) = calculate_roi_and_neuropil_traces(
        motion_corrected_stack, roi_mask_list, border
    )

    roi_file = Path(storage_directory) / "roi_traces.h5"
    write_trace_file(roi_traces, roi_names, roi_file)

    np_file = Path(storage_directory) / "neuropil_traces.h5"
    write_trace_file(neuropil_traces, roi_names, np_file)

    np_mask_file = Path(storage_directory) / "neuropil_masks.json"
    write_mask_file(neuropil_mask_list, roi_names, np_mask_file)

    return {
        "neuropil_trace_file": str(np_file),
        "roi_trace_file": str(roi_file),
        "neuropil_mask_file": str(np_mask_file),
        "exclusion_labels": exclusions,
    }
