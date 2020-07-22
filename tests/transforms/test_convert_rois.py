import pytest
import json
import os

from ophys_etl.transforms.convert_rois import (BinarizerAndROICreator,
                                               BinarizeAndCreationException)


@pytest.mark.parametrize("s2p_stat_fixture, ophys_movie_fixture, "
                         "motion_correction_fixture, binary_quantile, "
                         "abs_threshold, x_fail, file_remove, x_error",
                         [({}, {}, {}, -1, None,
                           True, None, BinarizeAndCreationException),
                          ({}, {}, {}, 0.1, -1,
                           True, None, BinarizeAndCreationException),
                          ({}, {}, {}, 0.1, None,
                           True, 'suite2p_stat_path',
                           BinarizeAndCreationException),
                          ({}, {}, {}, 0.1, None,
                           True, 'motion_corrected_video',
                           BinarizeAndCreationException)],
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
        converter = BinarizerAndROICreator(input_data=args,
                                           args=[])


@pytest.mark.parametrize("s2p_stat_fixture, ophys_movie_fixture, "
                         "motion_correction_fixture",
                         [({}, {}, {})],
                         indirect=["s2p_stat_fixture",
                                   "ophys_movie_fixture",
                                   "motion_correction_fixture"])
def test_binarize_and_convert_rois(s2p_stat_fixture, ophys_movie_fixture,
                                   motion_correction_fixture, tmp_path):
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
