from typing import List, Union, Tuple
import matplotlib.figure as mplt_fig
import pathlib
import numpy as np
import PIL.Image

from ophys_etl.modules.segmentation.utils.roi_utils import (
    roi_list_from_file)

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
    add_list_of_roi_boundaries_to_img)

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


def create_roi_v_background_grid(
        background_paths: Union[pathlib.Path, List[pathlib.Path]],
        background_names: Union[str, List[str]],
        roi_paths: Union[pathlib.Path, List[pathlib.Path]],
        roi_names: Union[str, List[str]],
        color_list: List[Tuple[int, int, int]],
        invalid_color: Tuple[int, int, int] = (255, 0, 0),
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

    color_list: List[Tuple[int, int, in]]
        List of RGB color tuples to cycle through when plotting
        ROIs. The number of colors does not have to match the
        number of ROI sets; the code will just cycle through
        the list of provided colors.

    invalid_color: Tuple[int, int, int]
        RGB color to use for invalid ROIs.
        (default = (255, 0, 0))

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
    figure = mplt_fig.Figure(figsize=(figsize_per*(n_roi+1),
                                      figsize_per*n_bckgd))

    axes = [figure.add_subplot(n_bckgd, n_roi+1, ii)
            for ii in range(1, 1+n_bckgd*(n_roi+1), 1)]

    roi_lists = []
    for this_roi_paths in roi_paths:
        roi = roi_list_from_file(this_roi_paths)
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

        axis = axes[i_bckgd*(n_roi+1)]
        img = background_array
        axis.imshow(img)
        axis.set_title(background_names[i_bckgd], fontsize=fontsize)
        for ii in range(0, img.shape[0], img.shape[0]//4):
            axis.axhline(ii, color='w', alpha=0.25)
        for ii in range(0, img.shape[1], img.shape[1]//4):
            axis.axvline(ii, color='w', alpha=0.25)

        for i_roi in range(n_roi):
            i_color = i_roi % len(color_list)
            this_color = color_list[i_color]
            this_roi = roi_lists[i_roi]

            valid_roi_list = [roi for roi in this_roi if roi.valid_roi]
            invalid_roi_list = [roi for roi in this_roi if not roi.valid_roi]

            axis = axes[i_bckgd*(n_roi+1)+i_roi+1]
            if i_bckgd == 0:
                axis.set_title(roi_names[i_roi], fontsize=fontsize)

            img = add_list_of_roi_boundaries_to_img(
                                            background_array,
                                            invalid_roi_list,
                                            color=invalid_color,
                                            alpha=1.0)

            img = add_list_of_roi_boundaries_to_img(
                                            img,
                                            valid_roi_list,
                                            color=this_color,
                                            alpha=1.0)

            axis.imshow(img)
            for ii in range(0, img.shape[0], img.shape[0]//4):
                axis.axhline(ii, color='w', alpha=0.25)
            for ii in range(0, img.shape[1], img.shape[1]//4):
                axis.axvline(ii, color='w', alpha=0.25)

    figure.tight_layout()
    return figure
