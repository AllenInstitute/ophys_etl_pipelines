import pandas as pd
import numpy as np
from collections import namedtuple

motion_border = namedtuple('motion_border', 'left right up down')


def get_max_correction_values(x_series: pd.Series, y_series: pd.Series,
                              max_shift: float = 30.0) -> motion_border:
    """
    Gets the max correction values in the cardinal directions from a series
    of correction values in the x and y directions
    Parameters
    ----------
    x_series: pd.Series:
        A series of movements in the x direction
    y_series: pd.Series:
        A series of movements in the y direction
    max_shift: float
        Maximum shift to allow when considering motion correction. Any
        larger shifts are considered outliers.

    Returns
    -------
    motion_border
        A named tuple containing the maximum correction values found during
        motion correction workflow step. Saved with the direction is the order
        [Left, Right, Up, Down] and also with names.

    """
    # validate if input columns match supported versions of motion correction
    # files
    if max_shift <= 0:
        raise ValueError(f"Max Shift input: {max_shift}. A positive input"
                         f"input for max shift is required.")

    # filter based out analomies based on maximum_shift
    x_no_outliers = x_series[(x_series >= -max_shift) &
                             (x_series <= max_shift)]
    y_no_outliers = y_series[(y_series >= -max_shift) &
                             (y_series <= max_shift)]
    # calculate max border shifts
    right_shift = np.max(-1 * x_no_outliers.min(), 0)
    left_shift = np.max(x_no_outliers.max(), 0)
    down_shift = np.max(-1 * y_no_outliers.min(), 0)
    up_shift = np.max(y_no_outliers.max(), 0)

    max_border = motion_border(left=left_shift, right=right_shift,
                               up=up_shift, down=down_shift)

    # check if all exist
    if np.any(np.isnan(np.array(max_border))):
        raise ValueError("One or more motion correction border directions "
                         "was found to be Nan, max motion border found: "
                         f"{max_border}, with max_shift {max_shift}")

    return max_border
