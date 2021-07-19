from typing import List, Dict, Union
import matplotlib.figure as mplt_fig
import pathlib
import numpy as np
import PIL.Image
import json
import copy

from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysROI)

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
    add_roi_boundaries_to_img,
    convert_keys)

from ophys_etl.modules.segmentation.qc_utils.video_utils import (
    scale_video_to_uint8)


def roi_list_from_file(file_path: pathlib.Path) -> List[OphysROI]:
    output_list = []
    with open(file_path, 'rb') as in_file:
        roi_data_list = json.load(in_file)
        roi_data_list = convert_keys(roi_data_list)
        for roi_data in roi_data_list:
            roi = OphysROI.from_schema_dict(roi_data)
            output_list.append(roi)
    return output_list


def create_roi_v_background_grid(
        background_paths: Union[pathlib.Path, List[pathlib.Path]],
        background_names: Union[str, List[str]],
        roi_paths: Union[pathlib.Path, List[pathlib.Path]],
        roi_names: Union[str, List[str]],
        attribute_name: str = 'filtered_hnc_Gaussian') -> mplt_fig.Figure:

    color_list = ((0, 255, 0),
                  (255, 128, 0),
                  (51, 255, 255),
                  (255, 51, 255))

    if isinstance(roi_paths, pathlib.Path):
        roi_paths = [roi_paths]
        if not isinstance(roi_names, str):
            raise RuntimeError('roi_paths was a single path; '
                               'roi_names must be a single str; '
                               f'got {roi_names} instead')
        roi_names = [roi_names]
    elif not isinstance(roi_names, list):
        raise RuntimeError('You passed in a list of ROI paths, but '
                           f'roi_names is {roi_names} '
                           f'of type {type(roi_names)}. '
                           'This must also be a list.')

    if len(roi_names) != len(roi_paths):
        raise RuntimeError(f'{len(roi_paths)} roi paths, but '
                           f'{len(roi_names)} roi names. '
                           'These numbers must be equal.')

    if isinstance(background_paths, pathlib.Path):
        background_paths = [background_paths]
        background_names = [background_names]

    n_bckgd = len(background_paths)  # rows
    n_roi = len(roi_paths)    # columns
    fontsize = 30
    figure = mplt_fig.Figure(figsize=(10*(n_roi+1), 10*n_bckgd))

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
            raise RuntimeError('Do not know how to parse background image file '
                               f'{this_bckgd}; must be either .png or .pkl')

        if this_bckgd.suffix == '.png':
            qtiles = np.quantile(background_array, (0.1, 0.999))
        else:
            qtiles = (0, background_array.max())
        background_array = scale_video_to_uint8(background_array,
                                                qtiles[0],
                                                qtiles[1])
                                                #0,
                                                #background_array.max())

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
            i_color = i_roi%len(color_list)
            this_color = color_list[i_color]
            this_roi = roi_lists[i_roi]
            axis = axes[i_bckgd*(n_roi+1)+i_roi+1]
            if i_bckgd == 0:
                axis.set_title(roi_names[i_roi], fontsize=fontsize)
            img = add_roi_boundaries_to_img(background_array,
                                            this_roi,
                                            color=this_color,
                                            alpha=1.0)

            axis.imshow(img)
            for ii in range(0, img.shape[0], img.shape[0]//4):
                axis.axhline(ii, color='w', alpha=0.25)
            for ii in range(0, img.shape[1], img.shape[1]//4):
                axis.axvline(ii, color='w', alpha=0.25)

    figure.tight_layout()
    return figure
