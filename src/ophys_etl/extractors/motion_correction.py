from collections import namedtuple

import numpy as np
import pandas as pd
from pathlib import Path

MotionBorder = namedtuple('MotionBorder', ['left', 'right', 'up', 'down'])


def get_max_correction_values(x_series: pd.Series, y_series: pd.Series,
                              max_shift: float = 30.0) -> MotionBorder:
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

    For deprecated implementation see:
    allensdk.internal.brain_observatory.roi_filter_utils.calculate_max_border

    Returns
    -------
    MotionBorder
        A named tuple containing the maximum correction values found during
        motion correction workflow step. Saved with the following direction
        order [left, right, up, down].

    """
    # take abs of max shift as we are considering both positive and negative
    # directions
    max_shift = abs(max_shift)

    # filter based out analomies based on maximum_shift
    x_no_outliers = x_series[(x_series >= -max_shift)
                             & (x_series <= max_shift)]
    y_no_outliers = y_series[(y_series >= -max_shift)
                             & (y_series <= max_shift)]
    # calculate max border shifts
    right_shift = np.max(-1 * x_no_outliers.min(), 0)
    left_shift = np.max(x_no_outliers.max(), 0)
    down_shift = np.max(-1 * y_no_outliers.min(), 0)
    up_shift = np.max(y_no_outliers.max(), 0)

    max_border = MotionBorder(left=left_shift, right=right_shift,
                              up=up_shift, down=down_shift)

    # check if all exist
    if np.any(np.isnan(np.array(max_border))):
        raise ValueError("One or more motion correction border directions "
                         "was found to be Nan, max motion border found: "
                         f"{max_border}, with max_shift {max_shift}")

    return max_border


def get_max_correction_from_file(
        input_csv: Path, max_shift: float = 30.0) -> MotionBorder:
    """

    Parameters
    ----------
    input_csv: Path
        Path to motion correction values for each frame stored in .csv format.
        This .csv file is expected to have a header row of either:
        ['framenumber','x','y','correlation','kalman_x', 'kalman_y'] or
        ['framenumber','x','y','correlation','input_x','input_y','kalman_x',
         'kalman_y','algorithm','type']
    max_shift: float
        Maximum shift to allow when considering motion correction. Any
        larger shifts are considered outliers.

    Returns
    -------
    motion_border
        A named tuple containing the maximum correction values found during
        motion correction workflow step. Saved with the following direction
        order [left, right, up, down].

    """
    motion_correction_df = pd.read_csv(input_csv)
    motion_border = get_max_correction_values(
        x_series=motion_correction_df['x'],
        y_series=motion_correction_df['y'],
        max_shift=max_shift)
    return motion_border
