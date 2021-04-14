import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import matplotlib.colors

from typing import Tuple, List

import h5py
import numpy as np
import PIL
import scipy.stats
from ophys_etl.modules.decrosstalk.ophys_plane import DecrosstalkingOphysPlane


def find_problematic_rois(ophys_plane: DecrosstalkingOphysPlane,
                          roi_flags: dict):
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

            unmixed_signal = qc_data[f'ROI/{roi.roi_id}/roi/unmixed/signal/trace'][()]
            unmixed_events = qc_data[f'ROI/{roi.roi_id}/roi/unmixed/signal/events'][()]
            unmixed_crosstalk = qc_data[f'ROI/{roi.roi_id}/roi/unmixed/crosstalk/trace'][()]

            unmixed_model = scipy.stats.linregress(unmixed_signal[unmixed_events],
                                                   unmixed_crosstalk[unmixed_events])

            metric = np.abs(100.0*unmixed_model.slope)
            if metric > threshold:
                problematic_rois.add(roi.roi_id)
            else:
                good_rois.add(roi.roi_id)

    return {'good': good_rois,
            'problematic': problematic_rois}


def get_avg_mixing_matrix(ophys_plane: DecrosstalkingOphysPlane):
    """
    Get the average mixing matrix for a plane specified by
    ophys_plane
    """
    avg_matrix = np.zeros((2,2), dtype=float)
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
                    subplot_spec: matplotlib.axes.Axes,
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
            max_img[:,:,jj] = raw_img[:,:]
        max_img[:,:,3] = 255

        # superimpose gridlines
        for ix in range(n_cols//4, n_cols, n_cols//4):
            for ic in range(3):
                v = max_img[:, ix, ic]
                new_val = (v//3 + 2*255//3).astype(int)
                max_img[:, ix, ic] = new_val

        for iy in range(n_rows//4, n_rows, n_rows//4):
            for ic in range(3):
                v = max_img[iy, :, ic]
                new_val = (v//3 + 2*255//3).astype(int)
                max_img[iy, :, ic] = new_val

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

        extent=(20, 20+n_cols,
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
                    if mask_matrix[iy,ix]:
                        nc += 1
                        cx += x0+ix
                        cy += y0+iy
                        for ic in range(3):
                            max_img_copies[0][y0+iy, x0+ix, ic] = qc_color[ic]
                            max_img_copies[2][y0+iy, x0+ix, ic] = ghost_color[ic]
            cx = cx//nc
            cy = cy//nc
            axes[1].text(cx,
                         n_rows-cy,
                         f'{roi.roi_id-roi_min}',
                         color=id_color,
                         fontsize=6)

        # plot the maximum projection images with the ROI masks added
        for jj in (0,2):
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
        msg += f'[[%.3e, %.3e]\n' % (avg_mixing_matrix[0, 0],
                                      avg_mixing_matrix[0, 1])
        msg += f'[%.3e, %.3e]]\n' % (avg_mixing_matrix[1, 0],
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
    Generate the figure showing ROIs superimposed over the
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
                        print('got ghost ',roi)
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
    mega_axis = plt.Subplot(fig, outer_grid[:,:])
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
    for ii in range(min(len(ophys_planes),4)):
        plot_plane_pair(ophys_planes[ii],
                        roi_to_flag,
                        outer_grid[ii],
                        fig)

    fig.savefig(figure_path)
    plt.close(fig)
    return None
