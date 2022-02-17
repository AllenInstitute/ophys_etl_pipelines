import pytest
import tempfile
import pathlib
import numpy as np
import pandas as pd

from ophys_etl.utils.motion_border import (
    get_max_correction_values,
    get_max_correction_from_file,
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
                                        down=-0.01), False, None),
                          ({"x": [22.0, 42.0, 11.0],
                            "y": [7.0, -4.0, 56.0]}, 30.0,
                           MotionBorder(left=22.0, right=-11.0,
                                        up=7.0, down=4.0),
                           False, None),
                          ({"x": [-22.0, 42.0, 3.0],
                            "y": [7.0, -14.0, 56.0]}, -10.0,
                           MotionBorder(left=3.0, right=-3.0,
                                        up=7.0, down=-7.0),
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


@pytest.fixture(scope='session')
def motion_csv_path_fixture(tmp_path_factory):
    """
    Write out an example CSV for testing motion border detection;
    will only contain 'x' and 'y' columns.
    """
    dir_path = tmp_path_factory.mktemp('motion_csv')
    file_path = tempfile.mkstemp(dir=dir_path, suffix='.csv')[1]
    file_path = pathlib.Path(file_path)
    with open(file_path, 'w') as out_file:
        out_file.write('x,y\n')
        for ii in range(10):
            out_file.write('%d,%d\n' % (ii-5, ii-7))
    yield file_path
    if file_path.exists():
        file_path.unlink()


@pytest.mark.parametrize(
        'max_shift, expected',
        [(2, MotionBorder(left=2, right=2, up=2, down=2)),
         (6, MotionBorder(left=4, right=5, up=2, down=6)),
         (22, MotionBorder(left=4, right=5, up=2, down=7))])
def test_get_max_correction_from_file(
        motion_csv_path_fixture,
        max_shift,
        expected):
    """
    Test method to read a MotionBorder from a file
    """
    actual = get_max_correction_from_file(
                    input_csv=motion_csv_path_fixture,
                    max_shift=max_shift)

    np.testing.assert_allclose(np.array(actual), np.array(expected))
