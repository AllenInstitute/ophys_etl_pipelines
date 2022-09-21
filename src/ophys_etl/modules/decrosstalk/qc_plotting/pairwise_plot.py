"""
This script contains the code necessary to generate the plots comparing
pairs of overlapping ROIs
"""

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.colors

import h5py
import numpy as np
import PIL
import pathlib
import itertools
import logging

from typing import Tuple, List, Dict, Optional

from matplotlib.colorbar import Colorbar

from ophys_etl.types import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import DecrosstalkingOphysPlane


from ophys_etl.modules.decrosstalk.qc_plotting.utils import add_gridlines


logger = logging.getLogger(__name__)


def get_roi_pixels(roi_list: List[OphysROI]) -> Dict[int, set]:
    """
    Take a list of OphysROIs and return a dict
    that maps roi_id to a set of (x,y) pixel coordinates
    corresponding to the masks of the ROIs

    Parameters
    ----------
    roi_list: List[OphysROI]

    Returns
    -------
    roi_pixel_dict: dict
        A dict whose keys are the ROI IDs of the ROIs in the input
        plane and whose values are sets of tuples. Each tuple is
        an (x, y) pair denoting a pixel in the ROI's mask
    """

    roi_pixel_dict = {}
    for roi in roi_list:
        roi_id = roi.roi_id
        grid = np.meshgrid(roi.x0+np.arange(roi.width, dtype=int),
                           roi.y0+np.arange(roi.height, dtype=int))
        mask_arr = roi.mask_matrix.flatten()
        x_coords = grid[0].flatten()[mask_arr]
        y_coords = grid[1].flatten()[mask_arr]
        roi_pixel_dict[roi_id] = set([(x, y)
                                      for x, y
                                      in zip(x_coords, y_coords)])
    return roi_pixel_dict


def find_overlapping_roi_pairs(roi_list_0: List[OphysROI],
                               roi_list_1: List[OphysROI]
                               ) -> List[Tuple[int, int, float, float]]:
    """
    Find all overlapping pairs from two lists of OphysROIs

    Parameters
    ----------
    roi_list_0: List[OphysROI]

    roi_list_1: List[OphysROI]

    Return:
    -------
    overlapping_pairs: list
        A list of tuples. Each tuple contains
        roi_id_0
        roi_id_0
        fraction of roi_id_0 that overlaps roi_id_1
        fraction of roi_id_1 that overlaps roi_id_0
    """

    pixel_dict_0 = get_roi_pixels(roi_list_0)
    pixel_dict_1 = get_roi_pixels(roi_list_1)

    overlapping_pairs = []

    roi_id_list_0 = list(pixel_dict_0.keys())
    roi_id_list_1 = list(pixel_dict_1.keys())

    for roi_pair in itertools.product(roi_id_list_0,
                                      roi_id_list_1):
        roi0 = pixel_dict_0[roi_pair[0]]
        roi1 = pixel_dict_1[roi_pair[1]]
        overlap = roi0.intersection(roi1)
        n = len(overlap)
        if n > 0:
            datum = (roi_pair[0], roi_pair[1], n/len(roi0), n/len(roi1))
            overlapping_pairs.append(datum)
    return overlapping_pairs


def plot_img_with_roi(axis: matplotlib.axes.Axes,
                      max_img: np.ndarray,
                      coord_mins: Tuple[int, int],
                      roi: Optional[OphysROI] = None,
                      roi_color: Optional[Tuple] = None) -> None:
    """
    Plot a maximum projection thumbnail, optionally with an ROI
    superimposed

    Parameters
    ----------
    axis: matplotlib.axes.Axes
        The axis in which to plot

    max_img: np.array
        The maximum projection thumbnail

    coord_mins: Tuple
        (xmin, ymin) of thumbnail

    roi: Optional[OphysROI]
        The ROI to superimpose on the thumbnail
        (if relevant)

    roi_color: Optional[Tuple]
        The RGB color of the roi
        (if relevant)

    Returns
    -------
    None
    """
    xmin = coord_mins[0]
    ymin = coord_mins[1]

    local_img = np.copy(max_img)
    axis.tick_params(which='both', axis='both',
                     left=0, bottom=0,
                     labelleft=0, labelbottom=0)
    for s in ('top', 'bottom', 'left', 'right'):
        axis.spines[s].set_visible(False)

    if roi is not None:
        mask_matrix = roi.mask_matrix
        for _ix in range(roi.width):
            ix = _ix+roi.x0-xmin
            for _iy in range(roi.height):
                iy = _iy+roi.y0-ymin
                if mask_matrix[_iy, _ix]:
                    for ic in range(3):
                        local_img[iy, ix, ic] = roi_color[ic]

    axis.imshow(local_img)

    return None


def generate_2d_histogram(data_x: np.ndarray,
                          data_y: np.ndarray,
                          x_bounds: Tuple[float, float],
                          y_bounds: Tuple[float, float],
                          x_label: Tuple[str, str],
                          y_label: Tuple[str, str],
                          title: str,
                          hist_axis: matplotlib.axes.Axes,
                          cbar_axis: matplotlib.axes.Axes) -> None:
    """
    Add the 2D histogram of trace values to the pairwise comparison
    plot

    Parameters
    ----------
    data_x: np.ndarray
        The trace array to plot on the x axis

    data_y: np.ndarray
        The trace array to plot on the y axis

    x_bounds: Tuple[float, float]
        The (min, max) values of the x axis

    y_bounds: Typle[float, float]
        The (min, max) values of the y axis

    x_label: Tuple[str, str]
        Zeroth element is the label of the x axis;
        First element is the hexadecimal color of that label

    y_label: Tuple[str, str]
        Zeroth element is the label of the y axis;
        First element is the hexadecimal color of that label

    title: str
        The title to be applied to the whole plot

    hist_axis: matplotlib.axes.Axes
        The axis on which to plot the 2D histogram

    cbar_axis: matplotlib.axes.Axes
        The axis on which to plot the colorbar associated with
        the histogram

    Returns
    -------
    None
    """

    # filter out NaNs
    valid = np.logical_and(np.logical_not(np.isnan(data_x)),
                           np.logical_not(np.isnan(data_y)))
    data_x = data_x[valid]
    data_y = data_y[valid]
    if len(data_x) == 0:
        return None

    # construct custom color map for the 2D histogram
    raw_cmap = plt.get_cmap('Blues')
    data = [raw_cmap(x) for x in np.arange(0.3, 0.95, 0.05)]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('custom_blue',
                                                               data)

    xmin = x_bounds[0]
    ymin = y_bounds[0]

    # subtracting of xmin for conversion to pixel
    # coordinates in the image
    data_x = data_x-xmin
    data_y = data_y-ymin

    xmax = x_bounds[1]-xmin
    ymax = y_bounds[1]-ymin

    nx = 25  # number of x pixels
    ny = 25
    dx = xmax/nx  # size of x pixels
    dy = ymax/ny

    # convert traces to pixel integers
    xx = np.round(data_x/dx).astype(int)
    xx = np.where(xx < nx, xx, nx-1)

    yy = np.round(data_y/dy).astype(int)
    yy = np.where(yy < ny, yy, ny-1)

    # find number of timesteps that fall in each pixel
    one_d_coords = xx*ny+yy
    unq, unq_ct = np.unique(one_d_coords, return_counts=True)
    xx = unq//ny
    yy = unq % ny

    hist = np.zeros((ny, nx), dtype=int)
    hist[yy, xx] = unq_ct
    hist = np.where(hist > 0, hist, np.NaN)

    img = hist_axis.imshow(hist, cmap=cmap,
                           origin='lower')

    # find reasonable tick labels for x axis
    log10_nx = np.floor(np.log10(dx))
    xticks = None
    while xticks is None or len(xticks) > 4:
        log10_nx += 1
        v = np.power(10, log10_nx)
        for tick_dx in np.arange(v, 5*v, v):
            xticks = np.arange(xmin, xmax+xmin, tick_dx)
            if len(xticks) <= 4:
                break

    xtick_labels = ['%d' % x for x in xticks]
    hist_axis.set_xticks((xticks-xmin)/dx)
    hist_axis.set_xticklabels(xtick_labels, fontsize=7)

    # find reasonable tick labels for y axis
    log10_ny = np.floor(np.log10(dy))
    yticks = None
    while yticks is None or len(yticks) > 4:
        log10_ny += 1
        v = np.power(10, log10_ny)
        for tick_dy in np.arange(v, 5*v, v):
            yticks = np.arange(ymin, ymax+ymin, tick_dy)
            if len(yticks) <= 4:
                break

    ytick_labels = ['%d' % y for y in yticks]
    hist_axis.set_yticks((yticks-ymin)/dy)
    hist_axis.set_yticklabels(ytick_labels, fontsize=7)

    hist_axis.set_xlabel(x_label[0], color=x_label[1], fontsize=10)
    hist_axis.set_ylabel(y_label[0], color=y_label[1], fontsize=10)

    hist_axis.set_title(title, fontsize=10)

    # plot color bar
    _ = Colorbar(mappable=img, ax=cbar_axis)
    cbar_axis.yaxis.set_ticks_position('left')

    # find reasonable tick labels for color bar
    hist_min = np.nanmin(hist)
    hist_max = np.nanmax(hist)
    log10_nhist = np.floor(np.log10(hist_max))-2
    hist_ticks = None
    while hist_ticks is None or len(hist_ticks) > 4:
        log10_nhist += 1
        v = np.power(10, log10_nhist)
        for tick_hist in np.arange(v, 5*v, v):
            _min = np.round(hist_min/tick_hist).astype(int)*tick_hist
            _max = np.round(hist_max/tick_hist).astype(int)*tick_hist
            hist_ticks = np.arange(_min, _max+1, tick_hist)
            hist_ticks = hist_ticks[np.where(hist_ticks < hist_max)]
            if len(hist_ticks) <= 4:
                break

    cbar_axis.yaxis.set_ticks(hist_ticks)

    return None


def get_img_thumbnails(roi0: OphysROI,
                       roi1: OphysROI,
                       max_img_0_full: np.ndarray,
                       max_img_1_full: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray,
                                  Tuple[int, int],
                                  Tuple[int, int]]:
    """
    Take two ROIs and two max projection images and return
    thumbnails of the max projection images trimmed to focus
    specifically on the ROIs

    Parameters
    ----------
    roi0: OphysROI

    roi1: OphysROI

    max_img_0_full: np.ndarray
        The full max projection image of the plane containing roi0

    max_img_1_full: np.ndarray
        The full max projection image of the plane containing roi1

    Returns
    -------
    np.ndarray:
        The thumbnail extracted from max_img_0_full

    np.ndarray:
        The thumbnail extracted from max_img_1_full

    Tuple[int, int]:
        (xmin, xmax) of the thumbnails in pixel space

    Tuple[int, int]:
        (ymin, ymax) of the thumbnails in pixel space
    """
    # get sensible bounds for the max projection thumbnail
    xmin = min(roi0.x0, roi1.x0)
    xmax = max(roi0.x0+roi0.width, roi1.x0+roi1.width)
    ymin = min(roi0.y0, roi1.y0)
    ymax = max(roi0.y0+roi0.height, roi1.y0+roi1.height)

    slop = 50  # ideal number of pixels beyond ROI to show
    dim = max(xmax-xmin+slop, ymax-ymin+slop)

    xmin = max(0, xmin-slop//2)
    ymin = max(0, ymin-slop//2)
    xmax = xmin+dim
    ymax = ymin+dim

    shape = max_img_0_full.shape

    # check that bounds do not exceed max image bounds
    if xmax > shape[1]:
        xmax = shape[1]
        xmin = xmax-dim
    if ymax > shape[0]:
        ymax = shape[0]
        ymin = ymax-dim

    # truncate minimum, if necessary
    if xmin < 0:
        xmin = 0
    if ymin < 0:
        ymin = 0

    # create max projection thumbnails
    max_img_0 = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=int)
    max_img_1 = np.zeros((ymax-ymin, xmax-xmin, 3), dtype=int)
    for ic in range(3):
        max_img_0[:, :, ic] = max_img_0_full[ymin:ymax, xmin:xmax]
        max_img_1[:, :, ic] = max_img_1_full[ymin:ymax, xmin:xmax]

    # add gridlines to max projection thumbnails
    max_img_0 = add_gridlines(max_img_0, 4)
    max_img_1 = add_gridlines(max_img_1, 4)

    return (max_img_0,
            max_img_1,
            (xmin, xmax),
            (ymin, ymax))


def get_nanmaxed_timestep(
        trace: np.ndarray) -> Tuple[int, Optional[float]]:
    """
    For a trace, subtract off the median. Then return the index
    and value of maximum of the median-subtracted trace.

    If the trace is all NaNs, return (-1, None)

    Parameters
    ----------
    trace: np.ndarray

    Returns
    -------
    maxdex: int
        timestep index of the maximum median-subtracted value

    maxval: Optional[float]
        maximum median subtracted value (None if trace is all NaNs)
    """
    if np.all(np.isnan(trace)):
        return (-1, None)
    mu = np.nanmedian(trace)
    maxdex = np.nanargmax(trace - mu)
    maxval = trace[maxdex] - mu
    return (maxdex, maxval)


def get_most_active_section(
        trace0: np.ndarray,
        trace1: np.ndarray,
        n_timesteps: int,
        n_ignore: int = 0) -> Tuple[int, int]:
    """
    Take two coupled traces, return the (i0, i1) timestep
    bounds of the segment of length n_timesteps with the
    largest maximum value for (trace-median_of_trace).

    Parameters
    ----------
    trace0: np.ndarray

    trace1: np.darray

    n_timesteps: int
        The length of the segment to be identified
        (in timesteps)

    n_ignore: int
        Ignore the first n_ignore timesteps when selecting
        the most active region

    Returns
    -------
    bounds: Tuple[int, int]
       Bounds such that trace0[bounds[0]:bounds[1]]
       yields the desired interval
    """

    # mask the first n_ignore timesteps with
    # np.NaN
    if n_ignore > 0:
        trace0 = np.copy(trace0)
        trace1 = np.copy(trace1)
        trace0[:n_ignore] = np.NaN
        trace1[:n_ignore] = np.NaN

    (maxdex0, max0) = get_nanmaxed_timestep(trace0)
    (maxdex1, max1) = get_nanmaxed_timestep(trace1)

    if max0 is None and max1 is None:
        return (0, min(n_timesteps, len(trace0)))
    elif max0 is None:
        chosendex = maxdex1
    elif max1 is None:
        chosendex = maxdex0
    elif max0 > max1:
        chosendex = maxdex0
    else:
        chosendex = maxdex1

    i0 = max(0, chosendex - n_timesteps//2)
    i1 = i0 + n_timesteps
    if i1 > len(trace0):
        i1 = len(trace0)
        i0 = max(0, i1-n_timesteps)

    return (i0, i1)


def plot_pair_of_rois(roi0: OphysROI,
                      roi1: OphysROI,
                      qc0: h5py.File,
                      qc1: h5py.File,
                      max_img_0_in: np.ndarray,
                      max_img_1_in: np.ndarray,
                      plotting_dir: pathlib.Path,
                      roi_pair: Tuple[int, int, float, float]):
    """
    Generate a set of plots comparing two ROIs from different planes
    which overlap in pixel space

    Parameters
    ----------
    roi0: OphysROI

    roi1: OphysROI

    qc0: h5py.File
        The QC data associated with roi0

    qc1: h5py.File
        The QC data associated with roi1

    max_img_0_in: np.ndarray
        The np.ndarray containig the max projection image for
        the plane in which roi0 exists

    max_img_1_in: np.ndarray
        The np.ndarray containig the max projection image for
        the plane in which roi1 exists

    plotting_dir: pathlib.Path
        The directory into which to place this figure

    roi_pair: Tuple[int, int, float, float]
        A tuple containing
        roi_id of roi0
        roi_id of roi1
        fraction of roi0 that overlaps roi1
        fraction of roi1 that overlaps roi0

    Returns
    -------
    None
        This method will generate a file
        {roi0.roi_id}_{roi1.roi_id}_comparison.png
        in plotting_dir
    """

    # find minimum ROI ID for each plane (necessary
    # for calculating abbreviated cell num)
    roi_keys = list(qc0['ROI'].keys())
    roi_id = np.array([int(k) for k in roi_keys])
    roi_min0 = roi_id.min()

    roi_keys = list(qc1['ROI'].keys())
    roi_id = np.array([int(k) for k in roi_keys])
    roi_min1 = roi_id.min()

    if max_img_0_in.shape != max_img_1_in.shape:
        raise RuntimeError("Problem in plot_pair_of_rois;\n"
                           "max projection image shapes do not align\n"
                           f"roi {roi0.roi_id}: {max_img_0_in.shape}\n"
                           f"roi {roi1.roi_id}:{max_img_1_in.shape}\n")

    color0 = matplotlib.colors.to_rgba('green')
    color0 = tuple(int(255*c) for c in color0[:3])

    color1 = matplotlib.colors.to_rgba('purple')
    color1 = tuple(int(255*c) for c in color1[:3])

    color0_hex = '#%02x%02x%02x' % color0[:3]
    color1_hex = '#%02x%02x%02x' % color1[:3]

    # If either ROI does not have a valid trace, do not
    # generate a plot
    valid_0 = qc0[f'ROI/{roi0.roi_id}/valid_unmixed_trace'][()]
    valid_1 = qc1[f'ROI/{roi1.roi_id}/valid_unmixed_trace'][()]
    if not valid_0:
        return None
    if not valid_1:
        return None

    (max_img_0,
     max_img_1,
     (xmin, xmax),
     (ymin, ymax)) = get_img_thumbnails(roi0,
                                        roi1,
                                        max_img_0_in,
                                        max_img_1_in)

    fig = plt.figure(figsize=(18, 4))
    axes = []  # list for subplots to be added to fig

    # using GridSpec to get granular control of each subplot's
    # size and location
    grid = gridspec.GridSpec(10, 37,
                             height_ratios=[10]*9+[3],
                             width_ratios=[10]*36+[5])
    grid.update(bottom=0.1, top=0.99, left=0.01, right=0.99,
                wspace=0.1, hspace=0.01)

    # axes for max projection images without ROI masks
    max0_axis = plt.Subplot(fig, grid[1:4, 0:3])
    max1_axis = plt.Subplot(fig, grid[6:9, 0:3])
    axes.append(max0_axis)
    axes.append(max1_axis)

    # axes for max projection images with ROI masks
    max0_roi_axis = plt.Subplot(fig, grid[1:4, 3:6])
    max1_roi_axis = plt.Subplot(fig, grid[6:9, 3:6])
    axes.append(max0_roi_axis)
    axes.append(max1_roi_axis)

    # axes for summary information about ROIs
    title0_axis = plt.Subplot(fig, grid[1:4, 0:6])
    title1_axis = plt.Subplot(fig, grid[6:9, 0:6])
    for ax in (title0_axis, title1_axis):
        ax.patch.set_alpha(0)
        ax.tick_params(which='both', axis='both',
                       left=0, bottom=0,
                       labelleft=0, labelbottom=0)

        for s in ('top', 'left', 'bottom', 'right'):
            ax.spines[s].set_visible(False)

    axes.append(title0_axis)
    axes.append(title1_axis)

    cell_num_0 = roi0.roi_id-roi_min0
    cell_num_1 = roi1.roi_id-roi_min1

    # fill out title_axes
    title0 = f"Plane {qc1['paired_plane'][()]};  roi {roi0.roi_id};  "
    title0 += f"cell num {cell_num_0}\n"
    title0 += "overlap: %.1f%%;      " % (roi_pair[2]*100)
    is_ghost = qc0[f'ROI/{roi0.roi_id}/is_ghost'][()]
    title0 += f"is_ghost: {is_ghost}"
    title0_axis.set_title(title0, fontsize=10,
                          horizontalalignment='left',
                          loc='left')

    title1 = f"Plane {qc0['paired_plane'][()]};  roi {roi1.roi_id};  "
    title1 += f"cell num {cell_num_1}\n"
    title1 += "Overlap: %.1f%%;      " % (roi_pair[3]*100)
    is_ghost = qc1[f'ROI/{roi1.roi_id}/is_ghost'][()]
    title1 += f"is_ghost: {is_ghost}"
    title1_axis.set_title(title1, fontsize=10,
                          horizontalalignment='left',
                          loc='left')

    # plot max projection thumbnails
    plot_img_with_roi(max0_axis,
                      max_img_0,
                      (xmin, ymin),
                      roi=None,
                      roi_color=None)

    plot_img_with_roi(max1_axis,
                      max_img_1,
                      (xmin, ymin),
                      roi=None,
                      roi_color=None)

    plot_img_with_roi(max0_roi_axis,
                      max_img_0,
                      (xmin, ymin),
                      roi=roi0,
                      roi_color=color0)

    plot_img_with_roi(max1_roi_axis,
                      max_img_1,
                      (xmin, ymin),
                      roi=roi1,
                      roi_color=color1)

    # Set params for controlling width of trace axes.
    # The main trace plot will extend from first_trace:last_trace
    # in the matplotlib GridSpec. The zoom-in on the most active
    # region will extend from last_trace+1:last_zoom
    #
    # There is not a well-motivated reason for these numbers beyond
    # "the main trace plot should be longer than the zoom-in trace plot,
    # and the zoom-in trace plot should be long enough that some details
    # are evident." This is the first combination that I tried. Natalia
    # was happy with the relative sizes of the plots.
    first_trace = 7
    last_trace = 24
    last_zoom = 30

    # number of timesteps to zoom in on for most active region
    n_zoom_timesteps = 1000

    raw_axis = plt.Subplot(fig, grid[1:4, first_trace:last_trace])
    unmixed_axis = plt.Subplot(fig, grid[6:9, first_trace:last_trace])
    axes.append(raw_axis)
    axes.append(unmixed_axis)

    raw_zoom_axis = plt.Subplot(fig, grid[1:4, last_trace+1:last_zoom])
    unmixed_zoom_axis = plt.Subplot(fig, grid[6:9, last_trace+1:last_zoom])
    axes.append(raw_zoom_axis)
    axes.append(unmixed_zoom_axis)

    trace_bounds = {}   # dict for storing max, min values of traces
    for trace_key in ("raw", "unmixed"):
        trace_bounds[trace_key] = {}
        for ii in (0, 1):
            trace_bounds[trace_key][ii] = {'min': None, 'max': None}

    # Communication with Natalia Orlova:
    # 'let's select activity in pre-decrosstalking and use the same time
    # interval for post-decrosstalking for the zoom-in since the point
    # here is to show "decrosstalking quality"'
    #
    # she also requested that we ignore the first 5000 timesteps
    # when constructing the zoom-in plot so that initial activity
    # spikes do not pollute the visualization

    raw_trace0 = qc0[f'ROI/{roi0.roi_id}/roi/raw/signal/trace'][()]
    raw_trace1 = qc1[f'ROI/{roi1.roi_id}/roi/raw/signal/trace'][()]

    zoom_bounds = get_most_active_section(
                        trace0=raw_trace0,
                        trace1=raw_trace1,
                        n_timesteps=n_zoom_timesteps,
                        n_ignore=5000)

    for (trace_axis,
         zoom_axis,
         trace_key) in zip((raw_axis, unmixed_axis),
                           (raw_zoom_axis, unmixed_zoom_axis),
                           ('raw', 'unmixed')):

        for ax in (trace_axis, zoom_axis):
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)

            ax.tick_params(axis='both', labelsize=7)

        trace0 = qc0[f'ROI/{roi0.roi_id}/roi/{trace_key}/signal/trace'][()]
        trace1 = qc1[f'ROI/{roi1.roi_id}/roi/{trace_key}/signal/trace'][()]

        if trace_key not in trace_bounds:
            trace_bounds[trace_key] = {}
            trace_bounds[trace_key][0] = {}
            trace_bounds[trace_key][1] = {}

        bounds0 = trace_bounds[trace_key][0]
        bounds1 = trace_bounds[trace_key][1]

        t0_max = np.nanmax(trace0)
        t0_min = np.nanmin(trace0)
        t1_max = np.nanmax(trace1)
        t1_min = np.nanmin(trace1)

        if not np.isnan(t0_max):
            if (bounds0['max'] is None or t0_max > bounds0['max']):
                trace_bounds[trace_key][0]['max'] = t0_max

        if not np.isnan(t0_min):
            if (bounds0['min'] is None or t0_min < bounds0['min']):
                trace_bounds[trace_key][0]['min'] = t0_min

        if not np.isnan(t1_max):
            if (bounds1['max'] is None or t1_max > bounds1['max']):
                trace_bounds[trace_key][1]['max'] = t1_max

        if not np.isnan(t1_min):
            if (bounds1['min'] is None or t1_min < bounds1['min']):
                trace_bounds[trace_key][1]['min'] = t1_min

        t = np.arange(len(trace0), dtype=int)
        trace_axis.plot(t, trace0, color=color0_hex, linewidth=1)
        trace_axis.plot(t, trace1, color=color1_hex, linewidth=1)

        zoom_t = t[zoom_bounds[0]:zoom_bounds[1]]
        zoom_axis.plot(zoom_t,
                       trace0[zoom_bounds[0]:zoom_bounds[1]],
                       color=color0_hex,
                       linewidth=1)

        zoom_axis.plot(zoom_t,
                       trace1[zoom_bounds[0]:zoom_bounds[1]],
                       color=color1_hex,
                       linewidth=1)

        if trace_key == 'raw':
            trace_title = 'Traces before decrosstalking'
            n_actual = zoom_bounds[1]-zoom_bounds[0]
            zoom_title = f'Most active {n_actual} timesteps'
        else:
            trace_title = 'Traces after decrosstalking'
            zoom_title = None
        trace_axis.set_title(trace_title, fontsize=10)

        if zoom_title is not None:
            zoom_axis.set_title(zoom_title, fontsize=10)

    # make sure raw and unmixed zoom axes have the same vertical
    # extent
    raw_ylim = raw_zoom_axis.get_ylim()
    unmixed_ylim = unmixed_zoom_axis.get_ylim()
    ylim = (min(raw_ylim[0], unmixed_ylim[0]),
            max(raw_ylim[1], unmixed_ylim[1]))
    unmixed_zoom_axis.set_ylim(ylim)
    raw_zoom_axis.set_ylim(ylim)

    # plot 2D histograms of trace values at individual timesteps
    raw_hist_axis = plt.Subplot(fig, grid[1:4, last_zoom+1:36])
    unmixed_hist_axis = plt.Subplot(fig, grid[6:9, last_zoom+1:36])
    axes.append(raw_hist_axis)
    axes.append(unmixed_hist_axis)

    # plot colorbar
    raw_cbar_axis = plt.Subplot(fig, grid[1:4, 36])
    unmixed_cbar_axis = plt.Subplot(fig, grid[6:9, 36])
    axes.append(raw_cbar_axis)
    axes.append(unmixed_cbar_axis)

    # clean up bounds
    for trace_key in ('raw', 'unmixed'):
        for ii in (0, 1):

            mx = trace_bounds[trace_key][ii]['max']
            mn = trace_bounds[trace_key][ii]['min']

            if mn is None:
                mn = 0
                trace_bounds[trace_key][ii]['min'] = mn

            if mx is None:
                mx = mn + 1
                trace_bounds[trace_key][ii]['max'] = mx

            if mx-mn < 1.0e-10:
                trace_bounds[trace_key][ii]['max'] = mn + 1

    sub_dir = 'roi/raw/signal/trace'
    generate_2d_histogram(qc0[f'ROI/{roi0.roi_id}/{sub_dir}'][()],
                          qc1[f'ROI/{roi1.roi_id}/{sub_dir}'][()],
                          (np.floor(trace_bounds['raw'][0]['min']),
                           np.ceil(trace_bounds['raw'][0]['max'])),
                          (np.floor(trace_bounds['raw'][1]['min']),
                           np.ceil(trace_bounds['raw'][1]['max'])),
                          (f'{roi0.roi_id}', color0_hex),
                          (f'{roi1.roi_id}', color1_hex),
                          'Trace v trace before decrosstalking',
                          raw_hist_axis, raw_cbar_axis)

    sub_dir = 'roi/unmixed/signal/trace'
    generate_2d_histogram(qc0[f'ROI/{roi0.roi_id}/{sub_dir}'][()],
                          qc1[f'ROI/{roi1.roi_id}/{sub_dir}'][()],
                          (np.floor(trace_bounds['unmixed'][0]['min']),
                           np.ceil(trace_bounds['unmixed'][0]['max'])),
                          (np.floor(trace_bounds['unmixed'][1]['min']),
                           np.ceil(trace_bounds['unmixed'][1]['max'])),
                          (f'{roi0.roi_id}', color0_hex),
                          (f'{roi1.roi_id}', color1_hex),
                          'Trace v trace after decrosstalking',
                          unmixed_hist_axis, unmixed_cbar_axis)

    for ax in axes:
        fig.add_subplot(ax)

    fname = f'cells_{cell_num_0}_{cell_num_1}'
    fname += f'_rois_{roi0.roi_id}_{roi1.roi_id}_comparison.png'
    out_name = plotting_dir/fname
    fig.savefig(out_name)
    plt.close(fig)
    return None


def generate_pairwise_figures(
        ophys_planes: List[Tuple[DecrosstalkingOphysPlane,
                                 DecrosstalkingOphysPlane]],
        qc_dir: str) -> None:
    """
    Generate the pairwise ROI comparison plots for sets of coupled planes

    Parameters
    ----------
    ophys_planes: List[Tuple[DecrosstalkingOphysPlane,
                             DecrosstalkingOphysPlane]]
        Each tuple contains a pair of coupled planes

    qc_dir: str
        The parent directory into which figures will be written

    Returns
    -------
    None
        For each pair of planes, this method will create a sub dir of qc_dir
        {ophys_experiment_id_0}_{ophys_experiment_id_1}_roi_pairs
        containing pngs for each individual overlapping pair of ROIs.
        These pngs will be named like
        cells_{cell_num_0}_{cell_num_1}_rois_{roi_id_0}_{roi_id_1}_comparison.png
        where values of cell_num_N refer to the abbreviated ID given to
        the ROIs in the full field-of-view QC plot.
    """

    for plane_pair in ophys_planes:

        if plane_pair[0].experiment_id > plane_pair[1].experiment_id:
            plane_pair = (plane_pair[1], plane_pair[0])

        msg = 'generating pair plots for '
        msg += f'{plane_pair[0].experiment_id}; '
        msg += f'{plane_pair[1].experiment_id}'
        logger.info(msg)

        # find the pairs of ROIs for which we must generate figures
        overlapping_rois = find_overlapping_roi_pairs(plane_pair[0].roi_list,
                                                      plane_pair[1].roi_list)

        if len(overlapping_rois) == 0:
            continue

        id0 = plane_pair[0].experiment_id
        id1 = plane_pair[1].experiment_id

        # create the directory into which the figures will be plotted
        pairwise_dir = pathlib.Path(qc_dir)/f'{id0}_{id1}_roi_pairs'
        if not pairwise_dir.exists():
            pairwise_dir.mkdir(parents=True)
        if not pairwise_dir.is_dir():
            msg = f'{pairwise_dir.resolve()}\nis not a directory'
            raise RuntimeError(msg)

        # read in the max projection image for plane 0
        raw_img = PIL.Image.open(plane_pair[0].maximum_projection_image_path,
                                 mode='r')
        n_rows = raw_img.size[0]
        n_cols = raw_img.size[1]
        max_img_0 = np.array(raw_img).reshape(n_rows, n_cols)

        # read in the max projection image for plane 1
        raw_img = PIL.Image.open(plane_pair[1].maximum_projection_image_path,
                                 mode='r')
        n_rows = raw_img.size[0]
        n_cols = raw_img.size[1]
        max_img_1 = np.array(raw_img).reshape(n_rows, n_cols)

        # create a dict mapping roi_id to OphysROI
        roi_from_id = {}
        for roi in plane_pair[0].roi_list:
            if roi.roi_id in roi_from_id:
                raise RuntimeError(f'{roi.roi_id} already in roi_from_id')
            roi_from_id[roi.roi_id] = roi
        for roi in plane_pair[1].roi_list:
            if roi.roi_id in roi_from_id:
                raise RuntimeError(f'{roi.roi_id} already in roi_from_id')
            roi_from_id[roi.roi_id] = roi

        with h5py.File(plane_pair[0].qc_file_path, 'r') as qc_data_0:
            with h5py.File(plane_pair[1].qc_file_path, 'r') as qc_data_1:
                for roi_pair in overlapping_rois:
                    roi0 = roi_from_id[roi_pair[0]]
                    roi1 = roi_from_id[roi_pair[1]]
                    plot_pair_of_rois(roi0, roi1,
                                      qc_data_0, qc_data_1,
                                      max_img_0, max_img_1,
                                      pairwise_dir,
                                      roi_pair)
    return None
