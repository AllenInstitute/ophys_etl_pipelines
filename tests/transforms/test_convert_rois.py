import pytest
import json
import numpy as np

from ophys_etl.transforms.convert_rois import BinarizerAndROICreator
from ophys_etl.types import DenseROI
from ophys_etl.schemas import DenseROISchema


def test_output_schema_element():
    """test that attempts to keep the TypedDict and subschema element in sync
    """
    s = DenseROI(
            id=1,
            x=23,
            y=34,
            width=128,
            height=128,
            valid_roi=True,
            mask_matrix=[[True, True], [False, True]],
            max_correction_up=12,
            max_correction_down=12,
            max_correction_left=12,
            max_correction_right=12,
            mask_image_plane=0,
            exclusion_labels=['small_size', 'motion_border'])

    # does this example have exactly the keys specified in DenseROI?
    assert set(list(s.keys())) == set(list(DenseROI.__annotations__.keys()))

    # can't really validate the above, but we can check against our
    # output schema
    # validate the object with a marshmallow load()
    subschema = DenseROISchema()
    subschema.load(s)
    assert subschema.dump(s) == s


@pytest.mark.parametrize("s2p_stat_fixture, ophys_movie_fixture, "
                         "motion_correction_fixture, "
                         "longest_edge_threshold, expected_rois",
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
                            ]},
                           {'movie_shape': (10, 5, 5)},
                           {'abs_value_bound': 0.25,
                            'required_x_values': [-0.3, 0.3],
                            'required_y_values': [-0.3, 0.3]}, 10,
                           [
                               {'id': 0,
                                'x': 0,
                                'y': 0,
                                'height': 2,
                                'width': 3,
                                'valid_roi': True,
                                'mask_matrix': np.array(
                                    [[True, True, True],
                                     [True, False, True]]).tolist(),
                                'max_correction_up': 0.3,
                                'max_correction_down': 0.3,
                                'max_correction_left': 0.3,
                                'max_correction_right': 0.3,
                                'mask_image_plane': 0,
                                'exclusion_labels': ["motion_border"],
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
                           ]),
                          ({'frame_shape': (5, 5),
                            'masks': [
                                np.array([[0, 0, 0, 0, 0],
                                          [0, 0.7, 0.85, 0.98, 0.67],
                                          [0, 0.85, 0.79, 0.82, 0.78],
                                          [0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0]]),
                                np.array([[0, 0.76, 0.85, 0, 0],
                                          [0, 0.88, 0.65, 0, 0],
                                          [0, 0.95, 0.78, 0, 0],
                                          [0, 0.98, 0.88, 0, 0],
                                          [0, 0, 0, 0, 0]]),
                                np.array([[0, 1, 1, 1, 1],
                                          [0, 1, 1, 0.83, 1],
                                          [0, 1, 0.78, 0.85, 1],
                                          [0, 1, 1, 1, 1],
                                          [0, 0, 0, 0, 0]]),
                                np.array([[0, 0, 0, 0, 0],
                                          [0.67, 0.89, 0, 0, 0],
                                          [0, 0, 0, 0, 0.87],
                                          [0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0]]),
                                np.array([[0.89, 0, 0, 0, 0],
                                          [0.79, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0],
                                          [0, 0.85, 0, 0, 0],
                                          [0, 0, 0, 0, 0]]),
                                np.array([[0.89, 0.54, 0, 0, 0],
                                          [0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0.74]]),
                                np.array([[0, 0, 0, 0, 0],
                                          [0, 1, 0.96, 0, 0],
                                          [0, 0.67, 0.87, 0, 0],
                                          [0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0]]),
                                np.array([[0, 0, 0, 0, 0],
                                          [0, 1, 0.85, 0, 0],
                                          [0, 0, 1, 0, 0],
                                          [0, 0, 0, 0, 0],
                                          [0, 0, 0, 0, 0]])]},
                           {'movie_shape': (10, 5, 5)},
                           {'abs_value_bound': 0,
                            'required_x_values': [-0.1, 0.1],
                            'required_y_values': [-0.1, 0.1]}, 3,
                           [
                               {'exclusion_labels': [],
                                'height': 2,
                                'id': 0,
                                'mask_image_plane': 0,
                                'mask_matrix': [[True, True],
                                                [False, True]],
                                'max_correction_down': 0.1,
                                'max_correction_up': 0.1,
                                'max_correction_left': 0.1,
                                'max_correction_right': 0.1,
                                'valid_roi': True,
                                'width': 2,
                                'x': 1,
                                'y': 1},
                               {'exclusion_labels': [],
                                'height': 2,
                                'id': 1,
                                'mask_image_plane': 0,
                                'mask_matrix': [[True, False],
                                                [False, True]],
                                'max_correction_down': 0.1,
                                'max_correction_up': 0.1,
                                'max_correction_left': 0.1,
                                'max_correction_right': 0.1,
                                'valid_roi': True,
                                'width': 2,
                                'x': 1,
                                'y': 1}
                           ])],
                         indirect=["s2p_stat_fixture",
                                   "ophys_movie_fixture",
                                   "motion_correction_fixture"])
def test_binarize_and_convert_rois(s2p_stat_fixture, ophys_movie_fixture,
                                   motion_correction_fixture,
                                   longest_edge_threshold,
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
        'npixel_threshold': 1,
        'longest_edge_threshold': longest_edge_threshold
    }

    converter = BinarizerAndROICreator(input_data=args,
                                       args=[])
    converter.binarize_and_create()

    # assert file exists
    assert output_path.exists()

    with open(output_path) as open_output:
        rois = json.load(open_output)

        assert expected_rois == rois
