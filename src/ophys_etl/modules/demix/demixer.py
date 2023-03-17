import logging
import os
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse


def _demix_point(
    source_frame: np.ndarray,
    mask_traces: np.ndarray,
    flat_masks: sparse,
    pixels_per_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Helper function to run demixing for single point in time for a
    source with overlapping traces.

    Parameters
    ==========
    source_frame: values of movie source at the single time point,
    unraveled in the x-y dimension (1d array of length HxW )
    flat_masks: 2d-array of binary masks unraveled in the x-y dimension
    mask traces: values of mask trace at single time point (1d-array of
        length n, where `n` is number of masks)
    pixels_per_mask: Number of pixels for each mask associated with
        trace (1d-array of length `n`)

    Returns
    =======
    Array of demixed trace values for each mask if all trace data is
    nonzero. Otherwise, returns None.
    """
    mask_weighted_trace = mask_traces * pixels_per_mask

    # Skip if there is zero signal anywhere in one of the traces
    if (mask_weighted_trace == 0).any():
        return None
    norm_mat = sparse.diags(pixels_per_mask / mask_weighted_trace, offsets=0)
    source_mat = sparse.diags(source_frame, offsets=0)
    source_mask_projection = flat_masks.dot(source_mat)
    weighted_masks = norm_mat.dot(source_mask_projection)
    # cast to dense numpy array for linear solver because solution is dense
    overlap = flat_masks.dot(weighted_masks.T).toarray()
    try:
        demix_traces = linalg.solve(overlap, mask_weighted_trace)
    except linalg.LinAlgError:
        logging.warning("Singular matrix, using least squares to solve.")
        x, _, _, _ = linalg.lstsq(overlap, mask_weighted_trace)
        demix_traces = x
    return demix_traces


def demix_time_dep_masks(
    raw_traces: np.ndarray,
    stack: np.ndarray,
    masks: np.ndarray,
    max_block_size: int = 1000,
) -> Tuple[np.ndarray, list]:
    """
    Demix traces of potentially overlapping masks extraced from a single
    2p recording.

    :param raw_traces: 2d array of traces for each mask, of dimensions
        (n, t), where `t` is the number of time points and `n` is the
        number of masks.
    :param stack: 3d array representing a 1p recording movie, of
        dimensions (t, H, W) or corresponding hdf5 dataset.
    :param masks: 3d array of binary roi masks, of shape (n, H, W),
        where `n` is the number of masks, and HW are the dimensions of
        an individual frame in the movie `stack`.
    :max_block_size: int representing maximum number of movie frames to read
        at a time (-1 for full length `t` of `stack`) (the default is 1000)
    :return: Tuple of demixed traces and whether each frame was skipped
        in the demixing calculation.
    """
    N, T = raw_traces.shape
    _, x, y = masks.shape
    P = x * y

    if max_block_size == -1:
        max_block_size = T
    elif max_block_size < 1:
        raise ValueError(
            "Invalid maximum block size {}. Must be strictly "
            "positive (>= 1), or -1 for full length block "
            "size.".format(max_block_size)
        )

    num_pixels_in_mask = np.sum(masks, axis=(1, 2))

    flat_masks = masks.reshape(N, P)
    flat_masks = sparse.csr_matrix(flat_masks)

    drop_frames = []
    demix_traces = np.zeros((N, T))

    for t in range(T):

        block_t = t % max_block_size
        if block_t == 0:  # load next block into memory and reshape
            block_T = np.min([(T - t), max_block_size])
            stack_block = stack[t : t + block_T].reshape(block_T, P)

        demixed_point = _demix_point(
            stack_block[block_t],
            raw_traces[:, t],
            flat_masks,
            num_pixels_in_mask,
        )
        if demixed_point is not None:
            demix_traces[:, t] = demixed_point
            drop_frames.append(False)
        else:
            drop_frames.append(True)
    return demix_traces, drop_frames


def plot_traces(
    raw_trace: np.ndarray,
    demix_trace: np.ndarray,
    roi_id: str,
    roi_ind: int,
    save_file: Union[str, Path],
) -> None:
    """Plot traces

    Parameters
    ----------
    raw_trace: np.ndarray
    demix_trace: np.ndarray
    roi_id: str
    roi_ind: int
    save_file: Union[str, Path]
    """
    fig, ax = plt.subplots()

    ax.plot(raw_trace, label="Fluoresence")
    ax.plot(demix_trace, label="Demixed")
    ax.set_title("ROI ID(%d) index (%d)" % (roi_id, roi_ind))
    ax.legend()
    plt.savefig(save_file)
    plt.close(fig)


def find_zero_baselines(traces: np.ndarray) -> np.ndarray:
    """
    Given a traces array, return a list of indices
    with zero baselines.

    Parameters
    -----------
    traces: np.ndarray

    Returns
    --------
    np.ndarray
        array of trace indices with zero baseline
    """
    means = traces.mean(axis=1)
    stds = traces.std(axis=1)
    return np.where((means < stds))


def plot_negative_baselines(
    raw_traces: np.ndarray,
    demix_traces: np.ndarray,
    mask_array: np.ndarray,
    roi_ids_mask: np.ndarray,
    plot_dir: Union[str, Path],
    ext: str = "png",
) -> Set[int]:
    """
    Plot negative baselines

    Parameters
    -----------
    raw_traces: np.ndarray
        raw traces for an array of ROIs
    demix_traces: np.ndarray
        demix_traces for an array of ROIs
    mask_array: np.ndarray
        array of 2D arrays of masks for the array of ROIs 
    roi_ids_mask: np.ndarray
        array of ROI IDs
    plot_dir: Union[str, Path]
        path to save plot
    ext: str
        extension of plot image format, e.g. png

    Returns
    -------
    Set[int]
        set of overlapping ROI indices 
    """
    N, T = raw_traces.shape
    _, x, y = mask_array.shape

    logging.debug("finding negative baselines")
    neg_inds = find_negative_baselines(demix_traces)[0]

    overlap_inds = set()
    logging.debug("detected negative baselines: %s", str(neg_inds))
    for roi_ind in neg_inds:
        os.mkdir(plot_dir)

        save_file = os.path.join(
            plot_dir, str(roi_ids_mask[roi_ind]) + "_negative." + ext
        )
        plot_traces(
            raw_traces[roi_ind],
            demix_traces[roi_ind],
            roi_ids_mask[roi_ind],
            roi_ind,
            save_file,
        )

        """ plot overlapping masks """
        save_file = os.path.join(
            plot_dir, str(roi_ids_mask[roi_ind]) + "_negative_masks." + ext
        )
        roi_overlap_inds = plot_overlap_masks_lengthOne(
            roi_ind, mask_array, save_file
        )

        overlap_inds.update(roi_overlap_inds)

    zero_inds = find_zero_baselines(demix_traces)[0]
    logging.debug("detected zero baselines: %s", str(zero_inds))
    overlap_inds.update(zero_inds)

    return list(overlap_inds)


def plot_negative_transients(
    raw_traces: np.ndarray,
    demix_traces: np.ndarray,
    mask_array: np.ndarray,
    roi_ids_mask: np.ndarray,
    plot_dir: Union[str, Path],
    ext: str = "png",
) -> List[int]:
    """Plot negative transients

    """
    N, T = raw_traces.shape
    _, x, y = mask_array.shape

    logging.debug("finding negative transients")
    trans_ind_list1 = [
        find_negative_transients_threshold(trace=demix_traces[n])
        for n in range(N)
    ]
    rois_with_trans1 = [i for i in range(N) if len(trans_ind_list1[i]) > 0]
    rois_with_trans = np.unique(rois_with_trans1)
    rois_with_trans = [
        r for r in rois_with_trans if len(trans_ind_list1[r][0]) > 0
    ]

    logging.debug("plotting negative transients")

    flat_masks = mask_array.reshape(N, x * y)
    overlap = flat_masks.dot(flat_masks.T)
    overlap ^= np.diag(np.diag(overlap))

    for roi_ind in rois_with_trans:

        """plot biggest negative transient of this roi"""
        trans_ind_list = trans_ind_list1[roi_ind]

        trans_ind_list = trans_ind_list[0]
        trans_list = []
        for i in trans_ind_list:
            if i > 100 and i < T - 100:
                trans_list.append(demix_traces[roi_ind, i - 100 : i + 100])
            elif i > 100 and i >= T - 100:
                trans_list.append(demix_traces[roi_ind, i - 100 :])
            else:
                trans_list.append(demix_traces[roi_ind, : i + 100])

        Ntrans = len(trans_list)
        biggest_trans = 0
        for i in range(1, Ntrans):
            if np.amin(trans_list[i]) < np.amin(trans_list[biggest_trans]):
                biggest_trans = i

        trans_ind = trans_ind_list[biggest_trans]

        if np.sum(overlap[roi_ind]) > 0:
            savefile = os.path.join(
                plot_dir,
                str(roi_ids_mask[roi_ind]) + "_transient_valid." + ext,
            )
            plot_transients(
                roi_ind,
                trans_ind,
                mask_array,
                raw_traces,
                demix_traces,
                savefile,
            )

            """ plot overlapping masks """
            savefile = os.path.join(
                plot_dir,
                str(roi_ids_mask[roi_ind]) + "_masks_valid." + ext,
            )
            plot_overlap_masks_lengthOne(roi_ind, mask_array, savefile)
    return rois_with_trans


def rolling_window(trace: np.ndarray, window: int = 500) -> np.ndarray:
    """

    :param trace:
    :param window:
    :return:
    """

    shape = trace.shape[:-1] + (trace.shape[-1] - window + 1, window)
    strides = trace.strides + (trace.strides[-1],)

    return np.lib.stride_tricks.as_strided(trace, shape=shape, strides=strides)


def find_negative_baselines(trace):
    means = trace.mean(axis=1)
    stds = trace.std(axis=1)
    return np.where((means + stds) < 0)


def find_negative_transients_threshold(
    trace, window=500, length=10, std_devs=3
):
    trace = np.pad(
        trace,
        pad_width=(window - 1, 0),
        mode="constant",
        constant_values=[np.mean(trace[:window])],
    )
    rolling_mean = np.mean(rolling_window(trace, window), -1)
    rolling_std = np.std(rolling_window(trace, window), -1)

    below_thresh = trace[window - 1 :] < rolling_mean - std_devs * rolling_std
    below_thresh = np.pad(
        below_thresh, pad_width=(window - 1, 0), mode="constant"
    )
    trans_length = np.sum(rolling_window(below_thresh, length), -1)
    trans_length = trans_length[window - length :]

    trans_ind = np.where(trans_length == length)

    return trans_ind


def plot_overlap_masks_lengthOne(
    roi_ind: np.ndarray,
    masks: np.ndarray,
    savefile: Union[str, Path] = None,
    weighted: bool = False,
) -> np.ndarray:

    masks = np.array(masks).astype(float)
    N, x, y = masks.shape
    if np.sum(masks[-1]) == x * y:
        masks = masks[:-1]
        N -= 1

    flat_masks = masks.reshape(N, x * y)
    masks_overlap = flat_masks.dot(flat_masks.T)

    # rois (k) that roi_ind overlaps with
    ind_plot = np.where(masks_overlap[roi_ind] > 0)[0]

    for i in ind_plot:  # rois that overlap with each roi k
        ind_k = np.where(masks_overlap[i] > 0)[0]
        ind_plot = np.concatenate((ind_plot, ind_k))

    ind_plot = np.unique(ind_plot)
    ind_plot = np.concatenate(([roi_ind], ind_plot[ind_plot != roi_ind]))

    plt.figure()
    color_list = ["b", "g", "r", "c", "m", "y", "k"]
    Ncol = len(color_list)
    for num, i in enumerate(ind_plot):
        mask_plot = masks[i]
        if not weighted:
            mask_plot = ((num % Ncol) + 1) * np.ma.array(
                masks[i], mask=(masks[i] == 0)
            )
            plt.imshow(
                mask_plot,
                clim=(1.0, Ncol + 1),
                cmap=colors.ListedColormap(color_list),
                alpha=0.5,
                interpolation="nearest",
            )

        elif weighted:
            mask_plot = np.ma.array(masks[i], mask=(masks[i] == 0))
            plt.imshow(
                mask_plot, cmap="gray_r", alpha=0.5, interpolation="nearest"
            )

        plt.text(
            np.mean(np.where(np.sum(mask_plot, axis=0))),
            np.mean(np.where(np.sum(mask_plot, axis=1))),
            str(i),
        )

    mask_tot = np.sum(masks[ind_plot, :, :], axis=0)
    mask_x = np.sum(mask_tot, axis=0)
    mask_y = np.sum(mask_tot, axis=1)

    plt.xlim((np.amin(np.where(mask_x)) - 5, np.amax(np.where(mask_x)) + 5))
    plt.ylim((np.amin(np.where(mask_y)) - 5, np.amax(np.where(mask_y)) + 5))
    plt.title("Masks")

    if savefile is not None:
        plt.savefig(savefile)
    plt.close()

    return ind_plot


def plot_transients(
    roi_ind: np.ndarray,
    t_trans: int,
    masks: np.ndarray,
    traces: np.ndarray,
    demix_traces: np.ndarray,
    savefile: Union[str, Path],
) -> None:

    masks = np.array(masks).astype(float)
    N, x, y = masks.shape
    _, Nt = traces.shape

    flat_masks = masks.reshape(N, x * y)
    masks_overlap = flat_masks.dot(flat_masks.T)

    ind_plot = np.where(masks_overlap[roi_ind, :] > 0)[
        0
    ]  # rois (k) that roi_ind overlaps with
    for i in ind_plot:  # rois that overlap with each roi k
        ind_k = np.where(masks_overlap[i, :] > 0)[0]
        ind_plot = np.concatenate((ind_plot, ind_k))

    ind_plot = np.unique(ind_plot)
    ind_plot = np.concatenate(([roi_ind], ind_plot[ind_plot != roi_ind]))

    if t_trans > 150 and t_trans < Nt - 150:
        plot_t = range(t_trans - 150, t_trans + 150)
    elif t_trans > 150 and t_trans >= Nt - 150:
        plot_t = range(t_trans - 150, Nt)
    else:
        plot_t = range(0, t_trans + 150)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    color_list = ["b", "g", "r", "c", "m", "y", "k"]
    Ncol = len(color_list)

    for num, i in enumerate(ind_plot):
        ax[0].plot(
            plot_t,
            traces[i, plot_t],
            label=str(i),
            color=color_list[(num % Ncol)],
        )
        ax[1].plot(
            plot_t,
            demix_traces[i, plot_t],
            label=str(i),
            color=color_list[(num % Ncol)],
        )

    ax[0].set_title("Raw")
    ax[0].set_ylabel("Fluorescence")
    ax[1].set_title("Demixed")
    ax[1].set_xlabel("Time")
    ax[0].legend(loc=0)

    plt.savefig(savefile)
    plt.close(fig)
