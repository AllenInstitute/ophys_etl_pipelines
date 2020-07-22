import pytest
import json
import os
import random

from ophys_etl.transforms.convert_rois import (BinarizerAndROICreator,
                                               BinarizeAndCreationException)


@pytest.mark.parametrize("s2p_stat_fixture, ophys_movie_fixture, "
                         "motion_correction_fixture, seed, binary_quantile, "
                         "abs_threshold, x_fail, file_remove, x_error",
                         [({}, {}, {}, 0, None, None,
                           False, None, None),
                          ({}, {}, {}, 0, 0.2, None,
                           False, None, None),
                          ({}, {}, {}, 0, -1, None,
                           True, None, BinarizeAndCreationException),
                          ({}, {}, {}, 0, None, -1,
                           True, None, BinarizeAndCreationException),
                          ({}, {}, {}, 0, None, None,
                           True, 'suite2p_stat_path',
                           BinarizeAndCreationException),
                          ({}, {}, {}, 0, None, None,
                           True, 'motion_corrected_video',
                           BinarizeAndCreationException)
                          ],
                         indirect=["s2p_stat_fixture",
                                   "ophys_movie_fixture",
                                   "motion_correction_fixture"])
def test_binarize_and_convert_rois(s2p_stat_fixture, ophys_movie_fixture,
                                   motion_correction_fixture, seed,
                                   binary_quantile, abs_threshold,
                                   x_fail, file_remove, x_error, tmp_path):
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
    if x_fail:
        with pytest.raises(x_error):
            if file_remove:
                os.remove(args[file_remove])
            converter = BinarizerAndROICreator(input_data=args,
                                               args=[])
            converter.binarize_and_create()
    else:
        converter = BinarizerAndROICreator(input_data=args,
                                           args=[])
        converter.binarize_and_create()

        # assert file exists
        assert output_path.exists()

        with open(output_path) as open_output:
            rois = json.load(open_output)
            # assert all rois were written
            assert len(rois) == len(stat_fixure_params['masks'])

            # assert expected data exists within random roi
            random.seed(seed)
            random_idx = random.randint(0, len(rois) - 1)
            expected_keys = {'x', 'y', 'height', 'width',
                             'max_correction_left',
                             'max_correction_right', 'max_correction_up',
                             'max_correction_down', 'id', 'cell_specimen_id',
                             'mask_matrix', 'valid_roi', 'exclusion_labels',
                             'mask_image_plane'}
            random_roi = rois[random_idx]
            assert set(random_roi.keys()) == expected_keys
