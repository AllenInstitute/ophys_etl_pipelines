import matplotlib
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.colors

import h5py
import numpy as np
import PIL
import pathlib
import scipy.stats
import itertools

from typing import Tuple, List, Dict, Optional

from matplotlib.colorbar import Colorbar

from ophys_etl.modules.decrosstalk.ophys_plane import DecrosstalkingOphysPlane
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

matplotlib.use('Agg')


def add_gridlines(img_array: np.ndarray,
                  denom: int) -> np.ndarray:
    """
    Add grid lines to an image

    Parameters
    ----------
    img_array: np.ndarray
        The input image

    denom: int
        1/alpha for the grid lines

    Returns
    -------
    out_array: np.ndarray
        The image with gridline superimposed
    """
    out_array = np.copy(img_array)
    nrows = img_array.shape[0]
    ncols = img_array.shape[1]
    for ix in range(nrows//4, nrows-4, nrows//4):
        for ic in range(3):
            v = out_array[ix, :, ic]
            new = ((denom-1)*v+255)//denom
            out_array[ix, :, ic] = new

    for iy in range(ncols//4, ncols-4, ncols//4):
        for ic in range(3):
            v = out_array[:, iy, ic]
            new = ((denom-1)*v+255)//denom
            out_array[:, iy, ic] = new

    return out_array


def find_problematic_rois(ophys_plane: DecrosstalkingOphysPlane,
                          roi_flags: dict) -> dict:
    """
    Figure out which ROIs have a reasonably small amount of crosstalk after
    demixing, and which do not

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
        'good' is a set of ROI ID indicating valid ROIs
        'problematic' is a set of ROI ID indicating problematic ROIs
    """
    good_rois = set()
    problematic_rois = set()

    threshold = 50.0

    h5_path = ophys_plane.qc_file_path
    with h5py.File(h5_path, 'r') as qc_data:
        for roi in ophys_plane.roi_list:
            if roi.roi_id in roi_flags:
                if roi_flags[roi.roi_id] == 'invalid':
                    continue
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
                problematic_rois.add(roi.roi_id)
            else:
                good_rois.add(roi.roi_id)

    return {'good': good_rois,
            'problematic': problematic_rois}


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

    green = matplotlib.colors.to_rgba('green')
    green = tuple([int(255*g) for g in green[:3]])

    red = matplotlib.colors.to_rgba('red')
    red = tuple([int(255*r) for r in red[:3]])

    outlier_color = (255, 108, 220)
    valid_color = (255, 196, 92)

    outlier_color_hex = '#%02x%02x%02x' % outlier_color
    valid_color_hex = '#%02x%02x%02x' % valid_color
    green_hex = '#%02x%02x%02x' % green
    red_hex = '#%02x%02x%02x' % red

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
        roi_qc = find_problematic_rois(plane, roi_flags)
        n_valid_roi = 0
        n_ghost_roi = 0

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
            ghost_color = green
            if roi.roi_id in roi_flags:
                if roi_flags[roi.roi_id] == 'invalid':
                    # if the ROI is globally invalid, move on
                    continue
                if roi_flags[roi.roi_id] == 'ghost':
                    ghost_color = red
                    n_ghost_roi += 1

            n_valid_roi += 1
            if roi.roi_id in roi_qc['problematic']:
                id_color = outlier_color_hex
                qc_color = outlier_color
            else:
                if roi.roi_id not in roi_qc['good']:
                    raise RuntimeError(f'ROI {roi.roi_id} not in either '
                                       '"good" or "problematic" set')
                id_color = valid_color_hex
                qc_color = valid_color

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
                        for ic in range(3):
                            max_img_copies[0][y0+iy,
                                              x0+ix,
                                              ic] = qc_color[ic]

                            max_img_copies[2][y0+iy,
                                              x0+ix,
                                              ic] = ghost_color[ic]
            cx = cx//nc
            cy = cy//nc
            axes[1].text(cx,
                         n_rows-cy,
                         f'{roi.roi_id-roi_min}',
                         color=id_color,
                         fontsize=6)

        # plot the maximum projection images with the ROI masks added
        for jj in (0, 2):
            axes[jj].imshow(max_img_copies[jj],
                            extent=extent)

        axes[0].text(10, 30+n_rows,
                     f'Exp: {plane.experiment_id}; min ROI: {roi_min}',
                     fontsize=10)

        for ax in axes:
            fig.add_subplot(ax)

        # summary statisitcs
        ax = plt.Subplot(fig, inner_grid[ii, 3])
        ax.patch.set_alpha(0)
        for s in ('top', 'bottom', 'left', 'right'):
            ax.spines[s].set_visible(False)
            ax.tick_params(which='both', axis='both',
                           left=0, bottom=0,
                           labelleft=0, labelbottom=0)

        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

        msg = ''
        msg += f'# of ROIs: {n_valid_roi}\n'
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
                color=red_hex,
                fontsize=10,
                verticalalignment='top')

        ax.text(10, 33,
                'not-ghost ROIS',
                color=green_hex,
                fontsize=10,
                verticalalignment='top')

        ax.text(10, 26,
                'possible outlier',
                color=outlier_color_hex,
                fontsize=10,
                verticalalignment='top')

        ax.text(10, 19,
                'no problem',
                color=valid_color_hex,
                fontsize=10,
                verticalalignment='top')

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
                        assert roi not in roi_to_flag
                        roi_to_flag[roi] = 'ghost'
                        print('got ghost ', roi)
                elif 'invalid' in k:
                    for roi in plane[k]:
                        assert roi not in roi_to_flag
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
    roi_id_0 = roi0.roi_id
    roi_id_1 = roi1.roi_id

    # If either ROI does not have a valid trace, do not
    # generate a plot
    valid_0 = qc0[f'ROI/{roi_id_0}/valid_unmixed_trace'][()]
    valid_1 = qc1[f'ROI/{roi_id_1}/valid_unmixed_trace'][()]
    if not valid_0:
        return None
    if not valid_1:
        return None

    # get sensible bounds for the max projection thumbnail
    xmin = min(roi0.x0, roi1.x0)
    xmax = max(roi0.x0+roi0.width, roi1.x0+roi1.width)
    ymin = min(roi0.y0, roi1.y0)
    ymax = max(roi0.y0+roi0.height, roi1.y0+roi1.height)

    slop = 50  # ideal number of pixels beyond ROI to show
    dim = max(roi0.width+slop,
              roi1.width+slop,
              roi0.height+slop,
              roi1.height+slop)

    xmin = max(0, xmin-slop//2)
    ymin = max(0, ymin-slop//2)
    xmax = xmin+dim
    ymax = ymin+dim

    shape = max_img_0_in.shape

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
        max_img_0[:, :, ic] = max_img_0_in[ymin:ymax, xmin:xmax]
        max_img_1[:, :, ic] = max_img_1_in[ymin:ymax, xmin:xmax]

    # add gridlines to max projection thumbnails
    max_img_0 = add_gridlines(max_img_0, 4)
    max_img_1 = add_gridlines(max_img_1, 4)

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

    # fill out title_axes
    title0 = f"Plane {qc1['paired_plane'][()]};  roi {roi0.roi_id};  "
    title0 += f"cell num {roi0.roi_id-roi_min0}\n"
    title0 += "overlap: %.1f%%;      " % (roi_pair[2]*100)
    is_ghost = qc0[f'ROI/{roi0.roi_id}/is_ghost'][()]
    title0 += f"is_ghost: {is_ghost}"
    title0_axis.set_title(title0, fontsize=10,
                          horizontalalignment='left',
                          loc='left')

    title1 = f"Plane {qc0['paired_plane'][()]};  roi {roi1.roi_id};  "
    title1 += f"cell num {roi1.roi_id-roi_min1}\n"
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

    # plot traces
    raw_axis = plt.Subplot(fig, grid[1:4, 7:30])
    unmixed_axis = plt.Subplot(fig, grid[6:9, 7:30])
    axes.append(raw_axis)
    axes.append(unmixed_axis)

    trace_bounds = {}   # dict for storing max, min values of traces

    for ax, trace_key in zip((raw_axis, unmixed_axis),
                             ('raw', 'unmixed')):

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

        if ('max' not in bounds0 or trace0.max() > bounds0['max']):
            trace_bounds[trace_key][0]['max'] = trace0.max()

        if ('min' not in bounds0 or trace0.min() < bounds0['min']):
            trace_bounds[trace_key][0]['min'] = trace0.min()

        if ('max' not in bounds1 or trace1.max() > bounds1['max']):
            trace_bounds[trace_key][1]['max'] = trace1.max()

        if ('min' not in bounds1 or trace1.min() < bounds1['min']):
            trace_bounds[trace_key][1]['min'] = trace1.min()

        t = np.arange(len(trace0), dtype=int)
        ax.plot(t, trace0, color=color0_hex, linewidth=1)
        ax.plot(t, trace1, color=color1_hex, linewidth=1)
        if trace_key == 'raw':
            title = 'Traces before decrosstalking'
        else:
            title = 'Traces after decrosstalking'
        ax.set_title(title, fontsize=10)

    # plot 2D histograms of trace values at individual timesteps
    raw_hist_axis = plt.Subplot(fig, grid[1:4, 31:36])
    unmixed_hist_axis = plt.Subplot(fig, grid[6:9, 31:36])
    axes.append(raw_hist_axis)
    axes.append(unmixed_hist_axis)

    # plot colorbar
    raw_cbar_axis = plt.Subplot(fig, grid[1:4, 36])
    unmixed_cbar_axis = plt.Subplot(fig, grid[6:9, 36])
    axes.append(raw_cbar_axis)
    axes.append(unmixed_cbar_axis)

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

    out_name = plotting_dir/f'{roi_id_0}_{roi_id_1}_comparison.png'
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
        {roi_id_0}_{roi_id_1}_comparison.png
    """

    for plane_pair in ophys_planes:

        if plane_pair[0].experiment_id > plane_pair[1].experiment_id:
            plane_pair = (plane_pair[1], plane_pair[0])

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
        raw_img = PIL.Image.open(plane_pair[0].maximum_projection_image_path)
        n_rows = raw_img.size[0]
        n_cols = raw_img.size[1]
        max_img_0 = np.array(raw_img).reshape(n_rows, n_cols)

        # read in the max projection image for plane 1
        raw_img = PIL.Image.open(plane_pair[1].maximum_projection_image_path)
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
