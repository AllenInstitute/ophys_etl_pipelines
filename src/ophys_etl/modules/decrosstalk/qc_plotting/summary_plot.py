"""
This script contains the code necessary to generate the plot which shows
all eight experimental planes from a single Ophys session with their ROIs
overlaid
"""

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.colors

import h5py
import numpy as np
import PIL
import scipy.stats

from typing import Tuple, List

from ophys_etl.types import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import DecrosstalkingOphysPlane
from ophys_etl.modules.decrosstalk.qc_plotting.utils import add_gridlines


def find_roi_colors(ophys_plane: DecrosstalkingOphysPlane,
                    roi_flags: dict) -> dict:
    """
    Filter the ROIs based on quality metrics.
    Determine which colors should be used to plot the ROIs

    Parameters
    ----------
    ophys_plane: DecrosstalkingOphysPlane
        The Ophys experimental plane whose ROIs we are inspecting

    roi_flags: dict
        A dict which maps ROI ID to final output.json flag (so that
        we can ignore ROIs which have been judged invalid already)

    Returns
    -------
    dict
        The dict will map roi_id to a tuple of two colors.

        The zeroth color will be an RGB representation of the color
        which should be used to plot the ROI's mask on the max projection
        image based on whether or not the ROI is a ghost/

        The first color will be an RGB representation of the color
        which should be used to mark the ROI's centroid on the max
        projection image based on the amount of crosstalk in the ROI's
        trace.

        Additionally the following keys will point to hexadecimal colors
        used for global statistics

        'ghost' -- the color of ghost ROIs
        'not_ghost' -- the color of ROIs that are not ghosts
        'expected' -- the color of ROIs with reasonable crosstalk values
        'outlier' -- the color of ROIs with significant crosstalk

    Note
    ----
    ROIs that are flagged as 'invalid' in roi_flags will not appear
    in the output dict
    """
    green = matplotlib.colors.to_rgba('green')
    green = tuple([int(255*g) for g in green[:3]])

    red = matplotlib.colors.to_rgba('red')
    red = tuple([int(255*r) for r in red[:3]])

    outlier_color = (255, 108, 220)
    valid_color = (255, 196, 92)

    out_dict = {}
    out_dict['ghost'] = red
    out_dict['not_ghost'] = green
    out_dict['expected'] = valid_color
    out_dict['outlier'] = outlier_color

    threshold = 50.0

    h5_path = ophys_plane.qc_file_path
    with h5py.File(h5_path, 'r') as qc_data:
        for roi in ophys_plane.roi_list:
            roi_colors = []
            if roi.roi_id in roi_flags:
                if roi_flags[roi.roi_id] == 'invalid':
                    continue
                if roi_flags[roi.roi_id] == 'ghost':
                    roi_colors.append(red)

            if len(roi_colors) == 0:
                roi_colors.append(green)

            unmixed_dir = f'ROI/{roi.roi_id}/roi/unmixed'
            unmixed_signal = qc_data[f'{unmixed_dir}/signal/trace'][()]
            unmixed_events = qc_data[f'{unmixed_dir}/signal/events'][()]
            unmixed_crosstalk = qc_data[f'{unmixed_dir}/crosstalk/trace'][()]

            active_signal = unmixed_signal[unmixed_events]
            active_crosstalk = unmixed_crosstalk[unmixed_events]
            unmixed_model = scipy.stats.linregress(active_signal,
                                                   active_crosstalk)

            metric = np.abs(100.0*unmixed_model.slope)
            if metric > threshold:
                roi_colors.append(outlier_color)
            else:
                roi_colors.append(valid_color)

            out_dict[roi.roi_id] = roi_colors

    return out_dict


def get_avg_mixing_matrix(ophys_plane: DecrosstalkingOphysPlane) -> np.ndarray:
    """
    Get the average mixing matrix for a plane specified by
    ophys_plane

    Parameters
    ----------
    ophys_plane: DecrosstalkingOphysPlane

    Returns
    -------
    np.array
        The average mixing matrix of the plane

    Notes
    -----
    This method will scan QC data file associated with the plane.
    As soon as it finds an ROI flagged as having used the average mixing
    matrix, it will return that matrix. If no such ROI exists, the method
    returns the mean of all of the ROI-specific mixing matrix in the plane.
    """
    avg_matrix = np.zeros((2, 2), dtype=float)
    n_m = 0
    with h5py.File(ophys_plane.qc_file_path, 'r') as qc_data:
        roi_keys = list(qc_data['ROI'].keys())
        for roi in roi_keys:
            if not qc_data[f'ROI/{roi}/valid_unmixed_trace'][()]:
                continue
            if not qc_data[f'ROI/{roi}/valid_unmixed_active_trace'][()]:
                continue
            local_m = qc_data[f'ROI/{roi}/roi/unmixed/mixing_matrix'][()]
            if not qc_data[f'ROI/{roi}/roi/unmixed/converged'][()]:
                return local_m
            avg_matrix += local_m
            n_m += 1
    return avg_matrix/n_m


def get_max_projection_image(plane: DecrosstalkingOphysPlane
                             ) -> np.ndarray:
    """
    Return the maximum projection image for a DecrosstalkingOphsPlane
    with gridlines superimposed

    Parameters
    ----------
    plane: DecrosstalkingOphysPlane

    Return
    ------
    np.ndarray
        Suitable for plt.imshow
    """
    # load the raw max projection image
    raw_img = PIL.Image.open(plane.maximum_projection_image_path)
    n_rows = raw_img.size[0]
    n_cols = raw_img.size[1]
    raw_img = np.array(raw_img).reshape(n_rows, n_cols)

    # convert to an RGBA image
    max_img = np.zeros((n_rows, n_cols, 4), dtype=int)

    for jj in range(3):
        max_img[:, :, jj] = raw_img[:, :]
    max_img[:, :, 3] = 255

    # superimpose gridlines
    max_img = add_gridlines(max_img, 3)

    return max_img


def plot_roi_mask(roi: OphysROI,
                  mask_imgs: List[np.ndarray],
                  mask_colors: List[Tuple[int]],
                  centroid_axis: matplotlib.axes.Axes,
                  centroid_color: Tuple[int],
                  min_roi_id: int) -> None:
    """
    Add the mask and centroid of an ROI to
    max projection image thumbnails

    Parameters
    ----------
    roi: OphysROI
        The ROI being plotted

    mask_imgs: List[np.ndarray]
        A list of thumbnails on which to plot the ROI mask

    mask_colors: List[Tuple[int]]
        List of RGB colors to use when plotting the ROI
        on the images in mask_imgs

    centroid_axis: matplotlib.axes.Axes
        The matplotlib axis of the plot where the ROI's
        centroid will be marked

    centroid_color: Tuple[int]
        The RGB color of the numeral used to mark
        the ROI's centroid

    min_roi_id: int
        The minimum roi_id value for the plane
        (this is used to reduce roi.roi_id to a visualizable
        value in the centroid plot)

    Returns
    -------
    None
        This method just adds the ROI to the provided maximum
        projection images and centroid axis
    """
    n_rows = mask_imgs[0].shape[0]
    # find centroid
    x0 = roi.x0
    y0 = roi.y0
    cx = 0
    cy = 0
    nc = 0
    mask_matrix = roi.mask_matrix
    for ix in range(roi.width):
        for iy in range(roi.height):
            if mask_matrix[iy, ix]:
                nc += 1
                cx += x0+ix
                cy += y0+iy
                for img, color in zip(mask_imgs, mask_colors):
                    for ic in range(3):
                        img[y0+iy,
                            x0+ix,
                            ic] = color[ic]

    cx = cx//nc
    cy = cy//nc
    centroid_axis.text(cx,
                       n_rows-cy,
                       f'{roi.roi_id-min_roi_id}',
                       color='#%02x%02x%02x' % centroid_color,
                       fontsize=6)
    return None


def plot_summary_statistics(color_lookup: dict,
                            avg_mixing_matrix: np.ndarray,
                            n_ghost_roi: int,
                            n_total_roi: int,
                            ax: matplotlib.axes.Axes) -> None:
    """
    Plot the summary statistics panel in the full session summary plot

    Parameters
    ----------
    color_lookup: dict
        A dict that maps 'ghost', 'not_ghost', 'outlier',
        and 'expected' to the relevant RGB colors (see output
        of find_roi_colors)

    avg_mixing_matrix: np.ndarray
        The average mixing matrix of the plane

    n_ghost_roi: int
        The number of ghost ROIs in the plane

    n_total_roi: int
        The number of total valid ROIs in the plane (i.e. ROIs that
        are not invalid due to NaNs in traces)

    ax: matplotlib.axes.Axes
        The axis in which to draw this plot

    Returns
    -------
    None
        This method just generates the summary plot
        in the given axis
    """

    ax.patch.set_alpha(0)
    for s in ('top', 'bottom', 'left', 'right'):
        ax.spines[s].set_visible(False)
        ax.tick_params(which='both', axis='both',
                       left=0, bottom=0,
                       labelleft=0, labelbottom=0)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    msg = ''
    msg += f'# of ROIs: {n_total_roi}\n'
    msg += f'# of ghost ROIs: {n_ghost_roi}\n'
    msg += '\nAvg mixing matrix\n'
    msg += '[[%.3f, %.3f]\n' % (avg_mixing_matrix[0, 0],
                                avg_mixing_matrix[0, 1])
    msg += '[%.3f, %.3f]]\n' % (avg_mixing_matrix[1, 0],
                                avg_mixing_matrix[1, 1])

    ax.text(10, 90, msg,
            fontsize=10,
            verticalalignment='top')

    ax.text(10, 40,
            'ghost ROIS',
            color='#%02x%02x%02x' % color_lookup['ghost'],
            fontsize=10,
            verticalalignment='top')

    ax.text(10, 33,
            'not-ghost ROIS',
            color='#%02x%02x%02x' % color_lookup['not_ghost'],
            fontsize=10,
            verticalalignment='top')

    ax.text(10, 26,
            'possible outlier',
            color='#%02x%02x%02x' % color_lookup['outlier'],
            fontsize=10,
            verticalalignment='top')

    ax.text(10, 19,
            'no problem',
            color='#%02x%02x%02x' % color_lookup['expected'],
            fontsize=10,
            verticalalignment='top')

    return None


def plot_plane_pair(ophys_planes: Tuple[DecrosstalkingOphysPlane,
                                        DecrosstalkingOphysPlane],
                    roi_flags: dict,
                    subplot_spec: matplotlib.gridspec.SubplotSpec,
                    fig) -> None:
    """
    Plot a single pair of Ophys planes in the larger plot created
    by generate_roi_figure

    Parameters
    ----------
    ophys_planes: Tuples[DecrosstalkingOphysPlane,
                         DecrosstalkingOphysPlane]
        The two coupled planes to be plotted

    roi_flags: dict
       A dict mapping roi_id to flags ('invalid' or 'ghost')

    subplot_spec: matplotlib.gridspec.SubplotSpec
        The section of the outer grid in which this pair of
        planes will be plotted

    fig: matplotlib.figure.Figure
        The figure in which we are plotting

    Returns
    -------
    None
    """

    inner_grid = gridspec.GridSpecFromSubplotSpec(2, 4,
                                                  subplot_spec=subplot_spec,
                                                  wspace=0.05, hspace=0.)

    # loop over the two planes
    for ii in range(2):
        plane = ophys_planes[ii]

        with h5py.File(plane.qc_file_path, 'r') as in_file:
            if 'ROI' not in in_file:
                # There were no ROIs in this plane; just move on
                continue
            roi_keys = list(in_file['ROI'].keys())

        roi_id = np.array([int(k) for k in roi_keys])
        roi_min = roi_id.min()

        avg_mixing_matrix = get_avg_mixing_matrix(plane)
        roi_color_lookup = find_roi_colors(plane, roi_flags)
        n_total_roi = 0
        n_ghost_roi = 0

        max_img = get_max_projection_image(plane)
        n_rows = max_img.shape[0]
        n_cols = max_img.shape[1]

        axes = []
        for jj in range(3):
            ax = plt.Subplot(fig, inner_grid[ii, jj])
            ax.set_xlim(0, 35+n_cols)
            ax.set_ylim(0, 35+n_rows)
            ax.patch.set_alpha(0)
            for s in ('top', 'bottom', 'left', 'right'):
                ax.spines[s].set_visible(False)
                ax.tick_params(which='both', axis='both',
                               left=0, bottom=0,
                               labelleft=0, labelbottom=0)
            axes.append(ax)

        extent = (20, 20+n_cols,
                  20, 20+n_rows)

        # create three separate copies of the max_img
        # so that we only have to loop over the ROIs once
        max_img_copies = {}
        max_img_copies[0] = np.copy(max_img)
        max_img_copies[2] = np.copy(max_img)

        axes[1].imshow(np.copy(max_img),
                       extent=extent)

        # loop over ROIS, adding centroids and masks to the copied
        # maximum projection images
        for roi in plane.roi_list:
            if roi.roi_id not in roi_color_lookup:
                continue

            if roi.roi_id in roi_flags:
                if roi_flags[roi.roi_id] == 'ghost':
                    n_ghost_roi += 1
            n_total_roi += 1

            plot_roi_mask(roi,
                          [max_img_copies[0],
                           max_img_copies[2]],
                          [roi_color_lookup[roi.roi_id][1],
                           roi_color_lookup[roi.roi_id][0]],
                          axes[1],
                          roi_color_lookup[roi.roi_id][1],
                          roi_min)

        # plot the maximum projection images with the ROI masks added
        for jj in (0, 2):
            axes[jj].imshow(max_img_copies[jj],
                            extent=extent)

        # Add header information to leftmost plot
        axes[0].text(10, 30+n_rows,
                     f'Exp: {plane.experiment_id}; min ROI: {roi_min}',
                     fontsize=10)

        # summary statisitcs
        ax = plt.Subplot(fig, inner_grid[ii, 3])
        axes.append(ax)

        plot_summary_statistics(roi_color_lookup,
                                avg_mixing_matrix,
                                n_ghost_roi,
                                n_total_roi,
                                ax)

        for ax in axes:
            fig.add_subplot(ax)

    return None


def generate_roi_figure(ophys_session_id: int,
                        ophys_planes: List[Tuple[DecrosstalkingOphysPlane,
                                                 DecrosstalkingOphysPlane]],
                        roi_flags: dict,
                        figure_path: str) -> None:
    """
    Generate the summary figure showing ROIs superimposed over the
    maximum projection image

    Parameters
    ----------
    ophys_session_id: int
        The ID of this session

    ophys_planes: List[Tuple[DecrosstalkingOphysPlane,
                             DecrosstalkingOphysPlane]]
        Each element of the list is a tuple containing a pair of
        coupled DecrosstalkingOphysPlanes

    roi_flags: dict
        The deserialized output.json produced by the pipeline

    figure_path: str
        Path to which the figure should be saved

    Returns
    -------
    None
    """

    roi_to_flag = {}
    for pair in roi_flags['coupled_planes']:
        for plane in pair['planes']:
            for k in plane.keys():
                if 'ghost' in k:
                    for roi in plane[k]:
                        if roi in roi_to_flag:
                            raise RuntimeError(f'{roi} already in roi_to_flag')
                        roi_to_flag[roi] = 'ghost'
                elif 'invalid' in k:
                    for roi in plane[k]:
                        if roi in roi_to_flag:
                            raise RuntimeError(f'{roi} already in roi_to_flag')
                        roi_to_flag[roi] = 'invalid'

    fig = plt.figure(figsize=(17.5, 10))

    outer_grid = gridspec.GridSpec(2, 2,
                                   wspace=0.01,
                                   hspace=0.05)

    outer_grid.update(left=0.01, right=0.99,
                      bottom=0.01, top=0.95)

    # create one all-encompassing axis for assigning
    # the plot title
    mega_axis = plt.Subplot(fig, outer_grid[:, :])
    for spine in ('top', 'right', 'left', 'bottom'):
        mega_axis.spines[spine].set_visible(False)

    mega_axis.tick_params(axis='both', which='both',
                          bottom=0,
                          left=0,
                          labelbottom=0,
                          labelleft=0)

    mega_axis.set_title(f'Ophys_session_id: {ophys_session_id}',
                        fontsize=12)

    fig.add_subplot(mega_axis)

    # create sub-axes on which to build the blue frames around each
    # plane pair
    axes = []
    for ii in range(4):
        ax = plt.Subplot(fig, outer_grid[ii])
        axes.append(ax)
        fig.add_subplot(ax)

    for ax in axes:

        for spine in ('top', 'right', 'left', 'bottom'):
            ax.spines[spine].set_color('steelblue')
            ax.spines[spine].set_linewidth(2)

        ax.tick_params(axis='both', which='both',
                       bottom=0,
                       left=0,
                       labelbottom=0,
                       labelleft=0)

    # actually plot ROIs
    for ii in range(min(len(ophys_planes), 4)):
        plot_plane_pair(ophys_planes[ii],
                        roi_to_flag,
                        outer_grid[ii],
                        fig)

    fig.savefig(figure_path)
    plt.close(fig)
    return None
