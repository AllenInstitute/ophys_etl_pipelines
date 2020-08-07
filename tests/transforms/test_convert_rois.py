import pytest
import json
import os

import numpy as np
from marshmallow.validate import ValidationError

from ophys_etl.transforms.convert_rois import BinarizerAndROICreator


@pytest.mark.parametrize("s2p_stat_fixture, ophys_movie_fixture, "
                         "motion_correction_fixture, binary_quantile, "
                         "abs_threshold, x_fail, file_remove, x_error",
                         [({}, {}, {}, -1, None,
                           True, None, ValidationError),
                          ({}, {}, {}, 0.1, -1,
                           True, None, ValidationError),
                          ({}, {}, {}, 0.1, 1.5,
                           True, None, ValidationError),
                          ({}, {}, {}, 1.5, 0.1,
                           True, None, ValidationError),
                          ({}, {}, {}, 0.1, None,
                           True, 'suite2p_stat_path',
                           ValidationError),
                          ({}, {}, {}, 0.1, None,
                           True, 'motion_corrected_video',
                           ValidationError)],
                         indirect=["s2p_stat_fixture",
                                   "ophys_movie_fixture",
                                   "motion_correction_fixture"])
def test_binarize_and_convert_rois_schema(s2p_stat_fixture,
                                          ophys_movie_fixture,
                                          motion_correction_fixture,
                                          binary_quantile, abs_threshold,
                                          x_fail, file_remove, x_error,
                                          tmp_path):
    stat_path, stat_fixure_params = s2p_stat_fixture
    movie_path, movie_fixture_params = ophys_movie_fixture
    motion_path, motion_fixture_params = motion_correction_fixture
    output_path = tmp_path / 'output.json'
    args = {
        'suite2p_stat_path': str(stat_path),
        'motion_corrected_video': str(movie_path),
        'motion_correction_values': str(motion_path),
        'output_json': str(output_path),
        'binary_quantile': binary_quantile,
        'abs_threshold': abs_threshold
    }
    with pytest.raises(x_error):
        if file_remove:
            os.remove(args[file_remove])
        BinarizerAndROICreator(input_data=args,
                               args=[])


@pytest.mark.parametrize("s2p_stat_fixture, ophys_movie_fixture, "
                         "motion_correction_fixture, expected_rois",
                         [({'frame_shape': (5, 5),
                            'masks': [
                             np.array([[0.67, 0.87, 0.98, 0, 0],
                                       [0.75, 0.52, 0.79, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]]),
                             np.array([[0, 0, 0, 0, 0],
                                       [0, 0, 0.64, 0.79, 0],
                                       [0, 0.57, 0.45, 0.91, 0],
                                       [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]]),
                             np.array([[0, 0, 0, 0, 0],
                                       [0, 0, 0.55, 0.43, 0],
                                       [0, 0, 0.68, 0.40, 0],
                                       [0, 0, 0.79, 0, 0],
                                       [0, 0, 0, 0, 0]])
                         ]}, {}, {'abs_value_bound': 0.25,
                                  'included_values_x': [-0.3, 0.3],
                                  'included_values_y': [-0.3, 0.3]}, [
                             {'id': 0,
                              'x': 0,
                              'y': 0,
                              'height': 2,
                              'width': 3,
                              'valid_roi': False,
                              'mask_matrix': np.array(
                                  [[True, True, True],
                                   [True, False, True]]).tolist(),
                              'max_correction_up': 0.3,
                              'max_correction_down': 0.3,
                              'max_correction_left': 0.3,
                              'max_correction_right': 0.3,
                              'mask_image_plane': 0,
                              'exclusion_labels': ["motion_border"]
                              },
                             {
                              'id': 1,
                              'x': 1,
                              'y': 1,
                              'height': 2,
                              'width': 3,
                              'valid_roi': True,
                              'mask_matrix': np.array(
                                  [[False, True, True],
                                   [True, False, True]]).tolist(),
                              'max_correction_up': 0.3,
                              'max_correction_down': 0.3,
                              'max_correction_left': 0.3,
                              'max_correction_right': 0.3,
                              'mask_image_plane': 0,
                              'exclusion_labels': []
                              },
                             {
                              'id': 2,
                              'x': 2,
                              'y': 1,
                              'height': 3,
                              'width': 2,
                              'valid_roi': True,
                              'mask_matrix': np.array(
                                  [[True, True],
                                   [True, False],
                                   [True, False]]).tolist(),
                              'max_correction_up': 0.3,
                              'max_correction_down': 0.3,
                              'max_correction_left': 0.3,
                              'max_correction_right': 0.3,
                              'mask_image_plane': 0,
                              'exclusion_labels': []
                              }
                         ])],
                         indirect=["s2p_stat_fixture",
                                   "ophys_movie_fixture",
                                   "motion_correction_fixture"])
def test_binarize_and_convert_rois(s2p_stat_fixture, ophys_movie_fixture,
                                   motion_correction_fixture,
                                   expected_rois, tmp_path):
    stat_path, stat_fixure_params = s2p_stat_fixture
    movie_path, movie_fixture_params = ophys_movie_fixture
    motion_path, motion_fixture_params = motion_correction_fixture
    output_path = tmp_path / 'output.json'

    args = {
        'suite2p_stat_path': str(stat_path),
        'motion_corrected_video': str(movie_path),
        'motion_correction_values': str(motion_path),
        'output_json': str(output_path),
    }

    converter = BinarizerAndROICreator(input_data=args,
                                       args=[])
    converter.binarize_and_create()

    # assert file exists
    assert output_path.exists()

    with open(output_path) as open_output:
        rois = json.load(open_output)
        # assert all rois were written
        assert len(rois) == len(stat_fixure_params['masks'])
        assert expected_rois == rois
