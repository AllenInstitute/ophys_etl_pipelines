import pytest
import numpy as np
import pandas as pd

from ophys_etl.extractors.motion_correction import (
    get_max_correction_values,
    MotionBorder)


@pytest.mark.parametrize("motion_correction_data, max_shift,"
                         "expected_motion_border, x_fail, expected_error",
                         [({"x": [None, None],
                            "y": [0.430, 0.321]}, 30.0,
                             None, True, ValueError),
                          ({"x": [0.430, 0.321],
                            "frame_number": [0, 1]}, 30.0,
                             None, True, KeyError),
                          ({"x": [31, -32],
                            "y": [0.54, -0.17]}, 30.0,
                             None, True, ValueError),
                          ({"x": [15],
                            "y": [12]}, 30.0,
                           MotionBorder(left=15, right=-15, up=12, down=-12),
                           False, None),
                          ({"x": [-15],
                            "y": [-12]}, 30.0,
                           MotionBorder(left=-15, right=15, up=-12, down=12),
                           False, None),
                          ({"x": [15],
                            "y": [12]}, -1,
                           None, True, ValueError),
                          ({"x": [15, 12, 0.5, 0.67, -2],
                            "y": [7, 15, 0.56, -2.3, 4]}, 30.0,
                           MotionBorder(left=15, right=2, up=15, down=2.3),
                           False, None),
                          ({"x": [0.42, 0.57, 0.36],
                            "y": [0.01, 0.52, 0.21]}, 0.67,
                           MotionBorder(left=0.57, right=-0.36, up=0.52,
                                        down=-0.01), False, None)])
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
    8. All float values less than abs val 1
    """
    motion_correction_df = pd.DataFrame.from_dict(motion_correction_data)
    if x_fail:
        with pytest.raises(expected_error):
            get_max_correction_values(motion_correction_df["x"],
                                      motion_correction_df["y"],
                                      max_shift=max_shift)
    else:
        calculated_border = get_max_correction_values(
            motion_correction_df["x"],
            motion_correction_df["y"],
            max_shift=max_shift)
        np.testing.assert_allclose(np.array(expected_motion_border),
                                   np.array(calculated_border))
