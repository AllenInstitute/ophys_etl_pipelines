import pandas as pd
import numpy as np
from collections import namedtuple

motion_border = namedtuple('motion_border', 'left right up down')


def get_max_correction_values(motion_corr_df: pd.DataFrame,
                              max_shift: float = 30.0) -> motion_border:
    """
    Gets the max correction values in the cardinal directions from
    a motion correction dataframe created during Motion Correction workflow
    step.
    Parameters
    ----------
    motion_corr_df: pd.DataFrame
        The dataframe that contains the motion correction value calculated
        during motion correction workflow step of ophys segmentation pipeline.
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
    input_column_set = set(motion_corr_df.columns)
    required_columns = {'x', 'y'}
    if not required_columns.intersection(input_column_set) == required_columns:
        raise KeyError("Required columns to compute max correction not in "
                       f"input dataset. Supplied columns: {input_column_set}, "
                       f"need columns {required_columns}")
    if max_shift <= 0:
        raise ValueError(f"Max Shift input: {max_shift}. A positive input"
                         f"input for max shift is required.")

    # filter based out analomies based on maximum_shift
    x_no_outliers = motion_corr_df["x"][(motion_corr_df["x"] >= -max_shift) &
                                        (motion_corr_df["x"] <= max_shift)]
    y_no_outliers = motion_corr_df["y"][(motion_corr_df["y"] >= -max_shift) &
                                        (motion_corr_df["y"] <= max_shift)]
    # calculate max border shifts
    right_shift = -1 * x_no_outliers.min()
    left_shift = x_no_outliers.max()
    down_shift = -1 * y_no_outliers.min()
    up_shift = y_no_outliers.max()

    max_border = motion_border(left=left_shift, right=right_shift,
                               up=up_shift, down=down_shift)

    # check if all exist
    if np.any(np.isnan(np.array(max_border))):
        raise ValueError("One or more motion correction border directions "
                         "was found to be Nan, max motion border found: "
                         f"{max_border}, with max_shift {max_shift}")

    return max_border
