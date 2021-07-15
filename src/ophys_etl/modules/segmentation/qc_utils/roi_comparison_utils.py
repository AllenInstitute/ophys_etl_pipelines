from typing import List, Dict, Union
import matplotlib.figure as mplt_fig
import pathlib
import numpy as np
import PIL.Image
import json
import copy

from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysROI,
    intersection_over_union)

from ophys_etl.modules.segmentation.graph_utils.conversion import (
    graph_to_img)

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
    add_roi_boundaries_to_img,
    convert_keys)

from ophys_etl.modules.segmentation.qc_utils.video_utils import (
    scale_video_to_uint8)


def find_iou_roi_matches(baseline_roi_list: List[OphysROI],
                         test_roi_list: List[OphysROI],
                         iou_threshold: float) -> Dict[str, List[OphysROI]]:
    """
    Parameters
    ----------
    baseline_roi_list: List[OphysROI]

    test_roi_list: List[OphysROI]

    iou_threshold: float

    Returns
    -------
    dict
        'unmatched_baseline'
        'unmatched_test'
        'matched_baseline'
        'matched_test'
    """

    baseline = copy.deepcopy(baseline_roi_list)
    test = copy.deepcopy(test_roi_list)

    matched_test = []
    matched_baseline = []
    unmatched_baseline = []

    for baseline_roi in baseline:
        candidate_indexes = []
        candidate_iou = []
        for ii, test_roi in enumerate(test):
            iou = intersection_over_union(baseline_roi, test_roi)
            if iou > iou_threshold:
                candidate_iou.append(iou)
                candidate_indexes.append(ii)
        if len(candidate_iou) > 0:
            best = np.argmax(candidate_iou)
            matched_test.append(test.pop(candidate_indexes[best]))
            matched_baseline.append(baseline_roi)
        else:
            unmatched_baseline.append(baseline_roi)

    return {'matched_test': matched_test,
            'matched_baseline': matched_baseline,
            'unmatched_test': test,
            'unmatched_baseline': unmatched_baseline}


def read_roi_list(file_path: pathlib.Path) -> List[OphysROI]:
    output_list = []
    with open(file_path, 'rb') as in_file:
        roi_data_list = json.load(in_file)
        roi_data_list = convert_keys(roi_data_list)
        for roi_data in roi_data_list:
            roi = OphysROI.from_schema_dict(roi_data)
            output_list.append(roi)
    return output_list


def create_roi_summary_fig(
        background_path: pathlib.Path,
        baseline_roi_path: pathlib.Path,
        test_roi_path: Union[pathlib.Path, List[pathlib.Path]],
        test_roi_names: Union[str, List[str]],
        iou_threshold: float,
        attribute_name: str = 'filtered_hnc_Gaussian') -> mplt_fig.Figure:


    if isinstance(test_roi_path, pathlib.Path):
        test_roi_path = [test_roi_path]
        if not isinstance(test_roi_names, str):
            raise RuntimeError('test_roi_path was a single path; '
                               'test_roi_names must be a single str; '
                               f'got {test_roi_names} instead')
        test_roi_names = [test_roi_names]
    elif not isinstance(test_roi_names, list):
        raise RuntimeError('You passed in a list of ROI paths, but '
                           f'test_roi_names is {test_roi_names} '
                           f'of type {type(test_roi_names)}. '
                           'This must also be a list.')

    if len(test_roi_names) != len(test_roi_path):
        raise RuntimeError(f'{len(test_roi_path)} roi paths, but '
                           f'{len(test_roi_names)} roi names. '
                           'These numbers must be equal.')

    if background_path.suffix == '.png':
        background_array = np.array(PIL.Image.open(background_path, 'r'))
    elif background_path.suffix == '.pkl':
        background_array = graph_to_img(background_path,
                                        attribute_name=attribute_name)
    else:
        raise RuntimeError('Do not know how to parse background image file '
                           f'{background}; must be either .png or .pkl')

    background_array = scale_video_to_uint8(background_array,
                                            0,
                                            background_array.max())

    background_rgb = np.zeros((background_array.shape[0],
                               background_array.shape[1],
                               3), dtype=np.uint8)
    for ic in range(3):
        background_rgb[:, :, ic] = background_array
    background_array = background_rgb
    del background_rgb

    n_columns = len(test_roi_path)
    n_rows = 4

    fontsize = 30
    figure = mplt_fig.Figure(figsize=(10*n_columns, 10*n_rows))
    axes = [figure.add_subplot(n_rows, n_columns, ii)
            for ii in range(1, 1+n_rows*n_columns, 1)]

    baseline_color = (0, 255, 0)
    this_color = (255, 128, 0)


    baseline_roi = read_roi_list(baseline_roi_path)
    baseline_img = add_roi_boundaries_to_img(background_array,
                                             baseline_roi,
                                             color=baseline_color,
                                             alpha=1.0)

    for i_column, (roi_path, roi_name) in enumerate(zip(test_roi_path,
                                                        test_roi_names)):
        these_roi = read_roi_list(roi_path)

        comparison = find_iou_roi_matches(baseline_roi,
                                          these_roi,
                                          iou_threshold)

        just_these_img = add_roi_boundaries_to_img(background_array,
                                                   these_roi,
                                                   color=this_color,
                                                   alpha=1.0)

        matches_img = add_roi_boundaries_to_img(
                          background_array,
                          comparison['matched_baseline'],
                          color=baseline_color,
                          alpha=1.0)

        matches_img = add_roi_boundaries_to_img(
                          matches_img,
                          comparison['matched_test'],
                          color=this_color,
                          alpha=1.0)

        misses_img = add_roi_boundaries_to_img(
                         background_array,
                         comparison['unmatched_baseline'],
                         color=baseline_color,
                         alpha=1.0)

        misses_img = add_roi_boundaries_to_img(
                         misses_img,
                         comparison['unmatched_test'],
                         color=this_color,
                         alpha=1.0)

        axes[i_column].imshow(just_these_img)
        axes[i_column].set_title(roi_name,
                                 fontsize=fontsize)
        axes[n_columns+i_column].imshow(baseline_img)
        axes[n_columns+i_column].set_title('baseline',
                                           fontsize=fontsize)
        axes[2*n_columns+i_column].imshow(matches_img)
        axes[2*n_columns+i_column].set_title(
                 f'matches at iou={iou_threshold:.2f}', fontsize=fontsize)
        axes[3*n_columns+i_column].imshow(misses_img)
        axes[3*n_columns+i_column].set_title(
                f'misses at iou={iou_threshold:.2f}', fontsize=fontsize)

    for ii in range(0,
                    background_array.shape[0],
                    background_array.shape[0]//4):
        for ax in axes:
            ax.axhline(ii, color='w', alpha=0.25)
    for ii in range(0,
                    background_array.shape[1],
                    background_array.shape[1]//4):
        for ax in axes:
            ax.axvline(ii, color='w', alpha=0.25)
    figure.tight_layout()
    return figure
