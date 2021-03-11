import logging
import h5py
import numpy as np
import os

from ophys_etl.modules.trace_extraction.roi_masks import (
        NeuropilMask, create_roi_mask_array, validate_mask, create_roi_masks)


def calculate_traces(stack, mask_list, block_size=1000):
    '''
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
    float[number masks][number frames]
        This is the average response for each Mask in each image frame
    '''

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
            reasons = ", ".join([item["exclusion_label_name"]
                                 for item in current_exclusions])
            logging.warning("unable to extract traces for mask "
                            "\"{}\": {} ".format(mask.label, reasons))
            continue

        if not isinstance(mask.mask, np.ndarray):
            mask.mask = np.array(mask.mask)
        mask_areas[i] = mask.mask.sum()

    # calculate traces
    for frame_num in range(0, num_frames, block_size):
        if frame_num % block_size == 0:
            logging.debug("frame " + str(frame_num) + " of " + str(num_frames))
        frames = stack[frame_num:frame_num+block_size]

        for i in range(len(mask_list)):
            if not valid_masks[i]:
                continue

            mask = mask_list[i]
            subframe = frames[:, mask.y:mask.y + mask.height,
                              mask.x:mask.x + mask.width]

            total = subframe[:, mask.mask].sum(axis=1)
            traces[i, frame_num:frame_num+block_size] = total / mask_areas[i]

    return traces, exclusions


def calculate_roi_and_neuropil_traces(movie_h5, roi_mask_list, motion_border):
    """ get roi and neuropil masks """

    # a combined binary mask for all ROIs (this is used to
    #   subtracted ROIs from annuli
    mask_array = create_roi_mask_array(roi_mask_list)
    combined_mask = mask_array.max(axis=0)

    logging.info("%d total ROIs" % len(roi_mask_list))

    # create neuropil masks for the central ROIs
    neuropil_masks = []
    for m in roi_mask_list:
        nmask = NeuropilMask.create_neuropil_mask(m, motion_border,
                                                  combined_mask,
                                                  "neuropil for " + m.label)
        neuropil_masks.append(nmask)

    num_rois = len(roi_mask_list)
    # read the large image stack only once
    combined_list = roi_mask_list + neuropil_masks

    with h5py.File(movie_h5, "r") as movie_f:
        stack_frames = movie_f["data"]

        logging.info("Calculating %d traces (neuropil + ROI) over %d "
                     "frames" % (len(combined_list), len(stack_frames)))
        traces, exclusions = calculate_traces(stack_frames, combined_list)

        roi_traces = traces[:num_rois]
        neuropil_traces = traces[num_rois:]

    return roi_traces, neuropil_traces, exclusions


def write_trace_file(data, names, path):
    logging.debug("Writing {}".format(path))

    utf_dtype = h5py.special_dtype(vlen=str)
    with h5py.File(path, 'w') as fil:
        fil["data"] = data
        fil.create_dataset("roi_names",
                           data=np.array(names).astype(np.string_),
                           dtype=utf_dtype)


def extract_traces(motion_corrected_stack, motion_border,
                   storage_directory, rois, log_0, **kwargs):

    # find width and height of movie
    with h5py.File(motion_corrected_stack, "r") as f:
        d = f["data"]
        h = d.shape[1]
        w = d.shape[2]

    # motion border
    border = [
        motion_border["x0"],
        motion_border["x1"],
        motion_border["y0"],
        motion_border["y1"]
    ]

    # create roi mask objects
    roi_mask_list = create_roi_masks(rois, w, h, border)
    roi_names = [roi.label for roi in roi_mask_list]

    # extract traces
    roi_traces, neuropil_traces, exclusions = \
        calculate_roi_and_neuropil_traces(
            motion_corrected_stack, roi_mask_list, border)

    roi_file = os.path.abspath(os.path.join(storage_directory,
                               "roi_traces.h5"))
    write_trace_file(roi_traces, roi_names, roi_file)

    np_file = os.path.abspath(os.path.join(storage_directory,
                                           "neuropil_traces.h5"))
    write_trace_file(neuropil_traces, roi_names, np_file)

    return {
        'neuropil_trace_file': np_file,
        'roi_trace_file': roi_file,
        'exclusion_labels': exclusions
    }
