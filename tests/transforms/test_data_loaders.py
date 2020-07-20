import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from ophys_etl.transforms.data_loaders import (get_max_correction_border,
                                               _get_max_correction_values,
                                               motion_border)


@pytest.mark.parametrize("motion_correction_data, expected_max_border, x_fail",
                         [({'framenumber': [0, 1, 2, 3, 4, 5],
                            'x': [-0.187, -0.555, -0.150,
                                  -0.159, 0.047, 0.964],
                            'y': [0.430, -1.096, -1.774,
                                  -2.773, -3.920, -4.633],
                            'correlation': [70002.9, 61688.8,
                                            48442.2, 39674.9,
                                            25735.2, 13894.7],
                            'kalman_x': [-0.001, -0.004,
                                         -0.006, -0.007,
                                         -0.007, 0.009],
                            'kalman_y': [-0.002, -0.009,
                                         -0.024, -0.055,
                                         -0.108, -0.182]},
                           motion_border(left=0.964, right=0.555, down=4.633,
                                         up=0.430),
                           False),
                          ({}, (), True)])  # doesn't matter file not written
def test_get_max_correction_border_from_file(motion_correction_data,
                                             expected_max_border,
                                             x_fail):
    motion_correction_df = pd.DataFrame.from_dict(motion_correction_data)
    with tempfile.TemporaryDirectory() as tempdir:
        temp_file_path = Path(tempdir) / 'test_rigid_motion_transform.csv'
        if x_fail:
            with pytest.raises(FileNotFoundError,
                               match="Motion Correction File Not Found"):
                get_max_correction_border(temp_file_path)
        else:
            motion_correction_df.to_csv(temp_file_path.as_posix(), index=True)
            correction_border = get_max_correction_border(temp_file_path)
            np.testing.assert_allclose(np.array(expected_max_border),
                                       np.array(correction_border))


@pytest.mark.parametrize("motion_correction_data, max_shift,"
                         "expected_motion_border, x_fail, expected_error",
                         [({'x': [None, None],
                            'y': [0.430, 0.321]}, 30.0,
                             None, True, ValueError),
                          ({'x': [0.430, 0.321],
                            'frame_number': [0, 1]}, 30.0,
                             None, True, KeyError),
                          ({'x': [31, -32],
                            'y': [0.54, -0.17]}, 30.0,
                             None, True, ValueError),
                          ({'x': [15],
                            'y': [12]}, 30.0,
                           motion_border(left=15, right=-15, up=12, down=-12),
                           False, None),
                          ({'x': [-15],
                            'y': [-12]}, 30.0,
                           motion_border(left=-15, right=15, up=-12, down=12),
                           False, None),
                          ({'x': [15],
                            'y': [12]}, -1,
                           None, True, ValueError),
                          ({'x': [15, 12, 0.5, 0.67, -2],
                            'y': [7, 15, 0.56, -2.3, 4]}, 30.0,
                           motion_border(left=15, right=2, up=15, down=2.3),
                           False, None)])
def test_get_max_correction_border(motion_correction_data, max_shift,
                                   expected_motion_border, x_fail,
                                   expected_error):
    """
    Test Cases:
    1. No values in one of required column
    2. Missing required column
    3. One column no values abs less than max shift
    4. One value per column
    5. One negative value per column
    6. Negative max shift
    7. Standard case with positive and negative values
    """
    motion_correction_df = pd.DataFrame.from_dict(motion_correction_data)
    if x_fail:
        with pytest.raises(expected_error):
            _get_max_correction_values(motion_correction_df,
                                       max_shift=max_shift)
    else:
        calculated_border = _get_max_correction_values(motion_correction_df,
                                                       max_shift=max_shift)
        np.testing.assert_allclose(np.array(expected_motion_border),
                                   np.array(calculated_border))
