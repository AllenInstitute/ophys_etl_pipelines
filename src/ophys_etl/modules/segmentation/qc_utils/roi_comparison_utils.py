from typing import List, Union, Tuple, Dict
import matplotlib.figure as mplt_fig
from matplotlib import cm as mplt_cm
import pathlib
import numpy as np
import PIL.Image
import networkx

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.utils.roi_utils import (
    ophys_roi_list_from_file,
    do_rois_abut)

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
    add_roi_boundary_to_img)

from ophys_etl.modules.segmentation.qc_utils.video_utils import (
    scale_video_to_uint8)


def _validate_paths_v_names(
        paths: Union[pathlib.Path, List[pathlib.Path]],
        names: Union[str, List[str]]) -> Tuple[List[pathlib.Path],
                                               List[str]]:
    """
    Validate that you have passed in the same number of file paths
    and plot names, casting them into lists if they are not already.

    Parameters
    ----------
    paths: Union[pathlib.Path, List[pathlib.Path]]

    names: Union[str, List[str]]

    Returns
    -------
    path_list: List[pathlib.Path]

    name_list: List[str]

    Notes
    -----
    The outputs will be single element lists if the inputs
    are singletons

    Raises
    ------
    RuntimeError if the number of paths and names are mismatched
    """

    if isinstance(paths, pathlib.Path):
        paths = [paths]
    if isinstance(names, str):
        names = [names]

    if len(paths) != len(names):
        msg = f'paths: {paths}\n'
        msg += f'names: {names}\n'
        msg += 'These must be the same shape'
        raise RuntimeError(msg)
    return paths, names


def get_roi_color_map(
        roi_list: List[OphysROI]) -> Dict[int, Tuple[int, int, int]]:
    """
    Take a list of OphysROI and return a dict mapping ROI ID
    to RGB color so that no ROIs that touch have the same color

    Parametrs
    ---------
    roi_list: List[OphysROI]

    Returns
    -------
    color_map: Dict[int, Tuple[int, int, int]]
    """
    roi_graph = networkx.Graph()
    for roi in roi_list:
        roi_graph.add_node(roi.roi_id)
    for ii in range(len(roi_list)):
        roi0 = roi_list[ii]
        for jj in range(ii+1, len(roi_list)):
            roi1 = roi_list[jj]

            # value of 5 is so that singleton ROIs that
            # are near each other do not get assigned
            # the same color
            abut = do_rois_abut(roi0, roi1, 5.0)
            if abut:
                roi_graph.add_edge(roi0.roi_id, roi1.roi_id)
                roi_graph.add_edge(roi1.roi_id, roi0.roi_id)

    nx_coloring = networkx.greedy_color(roi_graph)
    n_colors = len(set(nx_coloring.values()))

    mplt_color_map = mplt_cm.jet

    # create a list of colors based on the matplotlib color map
    raw_color_list = []
    for ii in range(n_colors):
        color = mplt_color_map((1.0+ii)/(n_colors+1.0))
        color = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        raw_color_list.append(color)

    # re-order colors so that colors that are adjacent in index
    # have higher contrast
    step = max(n_colors//3, 1)
    color_list = []
    for i0 in range(step):
        for ii in range(i0, n_colors, step):
            this_color = raw_color_list[ii]
            color_list.append(this_color)

    # reverse color list, since matplotlib.cm.jet will
    # assign a dark blue as color_list[0], which isn't
    # great for contrast
    color_list.reverse()

    color_map = {}
    for roi_id in nx_coloring:
        color_map[roi_id] = color_list[nx_coloring[roi_id]]
    return color_map


def create_roi_v_background_grid(
        background_paths: Union[pathlib.Path, List[pathlib.Path]],
        background_names: Union[str, List[str]],
        roi_paths: Union[pathlib.Path, List[pathlib.Path]],
        roi_names: Union[str, List[str]],
        attribute_name: str = 'filtered_hnc_Gaussian',
        figsize_per: int = 10) -> mplt_fig.Figure:
    """
    Create a plot showing a set of ROIs overlaid over a set of
    different background images. In the final plot, each distinct
    background image will be a different row of subplots and each
    distinct set of ROIs will be a diferent column of subplots.

    Parameters
    ----------
    background_paths: Union[pathlib.Path, List[pathlib.Path]]
        Path(s) to file(s) containing background images. May be either
        PNG images or pkl files containing networkx graphs

    background_names: Union[str, List[str]]
       The names of the backgrounds as they will appear in the plot
       (there must be an equal number of background_names as
       background_paths)

    roi_paths: Union[pathlib.Path, List[pathlib.Path]]
        Path(s) to file(s) containing JSONized ROIs

    roi_names: Union[str, List[str]]
        The names of the ROI sets as they will appear in te plot
        (there must be an equal number of roi_names as roi_paths)

    attribute_name: str
        The name of the attribute to use in constructing the background
        image from a networkx graph, if applicable.
        Default: 'filtered_hnc_Gaussian'

    figsize_per: int
        When setting figsize for the output figure, each dimension of
        each subplot will be given this many inches (i.e.
        matplotlib.figure.Figure will be instantiated with
        figsize=(figsize_per*n_columns, figsize_per*n_rows)
        (default=10)

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    PNG background images will be clipped at the 0.1 and 0.999
    brightness quantiles.

    Raises
    ------
    RuntimeError if number of paths and names mismatch, either for ROIs
    or backgrounds.

    RuntimeError if a background_path does not end in '.png' or '.pkl'
    """

    (roi_paths,
     roi_names) = _validate_paths_v_names(roi_paths,
                                          roi_names)

    (background_paths,
     background_names) = _validate_paths_v_names(background_paths,
                                                 background_names)

    n_bckgd = len(background_paths)  # rows
    n_roi = len(roi_paths)    # columns
    fontsize = int(np.round(30.0*0.1*figsize_per))
    figure = mplt_fig.Figure(figsize=(figsize_per*(2*n_roi+1),
                                      figsize_per*n_bckgd))

    axes = [figure.add_subplot(n_bckgd, 2*n_roi+1, ii)
            for ii in range(1, 1+n_bckgd*(2*n_roi+1), 1)]

    roi_lists = []
    roi_color_maps = []
    for this_roi_paths in roi_paths:
        roi = ophys_roi_list_from_file(this_roi_paths)
        color_map = get_roi_color_map(roi)
        roi_color_maps.append(color_map)
        roi_lists.append(roi)

    for i_bckgd in range(n_bckgd):
        this_bckgd = background_paths[i_bckgd]
        if this_bckgd.suffix == '.png':
            background_array = np.array(PIL.Image.open(this_bckgd, 'r'))
        elif this_bckgd.suffix == '.pkl':
            background_array = graph_to_img(this_bckgd,
                                            attribute_name=attribute_name)
        else:
            raise RuntimeError('Do not know how to parse '
                               'background image file '
                               f'{this_bckgd}; must be '
                               'either .png or .pkl')

        if this_bckgd.suffix == '.png':
            qtiles = np.quantile(background_array, (0.1, 0.999))
        else:
            qtiles = (0, background_array.max())
        background_array = scale_video_to_uint8(background_array,
                                                qtiles[0],
                                                qtiles[1])

        background_rgb = np.zeros((background_array.shape[0],
                                   background_array.shape[1],
                                   3), dtype=np.uint8)
        for ic in range(3):
            background_rgb[:, :, ic] = background_array
        background_array = background_rgb
        del background_rgb

        axis = axes[i_bckgd*(2*n_roi+1)]
        img = background_array
        axis.imshow(img)
        axis.set_title(background_names[i_bckgd], fontsize=fontsize)
        for ii in range(0, img.shape[0], img.shape[0]//4):
            axis.axhline(ii, color='w', alpha=0.25)
        for ii in range(0, img.shape[1], img.shape[1]//4):
            axis.axvline(ii, color='w', alpha=0.25)

        for i_roi in range(n_roi):
            this_roi_list = roi_lists[i_roi]
            this_color_map = roi_color_maps[i_roi]

            valid_roi_list = [roi for roi in this_roi_list
                              if roi.valid_roi]

            invalid_roi_list = [roi for roi in this_roi_list
                                if not roi.valid_roi]

            valid_axis = axes[i_bckgd*(2*n_roi+1)+2*i_roi+1]
            invalid_axis = axes[i_bckgd*(2*n_roi+1)+2*i_roi+2]
            if i_bckgd == 0:
                valid_axis.set_title(f'{roi_names[i_roi]} (valid)',
                                     fontsize=fontsize)
                invalid_axis.set_title(f'{roi_names[i_roi]} (invalid)',
                                       fontsize=fontsize)

            valid_img = np.copy(background_array)
            for roi in valid_roi_list:
                valid_img = add_roi_boundary_to_img(
                                valid_img,
                                roi,
                                this_color_map[roi.roi_id],
                                1.0)
            invalid_img = np.copy(background_array)

            for roi in invalid_roi_list:
                invalid_img = add_roi_boundary_to_img(
                                 invalid_img,
                                 roi,
                                 this_color_map[roi.roi_id],
                                 1.0)

            valid_axis.imshow(valid_img)
            invalid_axis.imshow(invalid_img)
            for axis, img in zip([invalid_axis, valid_axis],
                                 [valid_img, invalid_img]):
                for ii in range(0, img.shape[0], img.shape[0]//4):
                    axis.axhline(ii, color='w', alpha=0.25)
                for ii in range(0, img.shape[1], img.shape[1]//4):
                    axis.axvline(ii, color='w', alpha=0.25)

    figure.tight_layout()
    return figure
