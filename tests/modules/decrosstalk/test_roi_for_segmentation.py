import pytest
import numpy as np
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI


@pytest.fixture
def roi_boundary_data():
    """
    Mask matrices and their expected boundaries
    """

    output = []

    # simple rectangle
    mask = [[False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False]]

    bdry = [[False, True, True, True, False],
            [False, True, False, True, False],
            [False, True, False, True, False],
            [False, True, True, True, False]]

    output.append({'mask_matrix': mask, 'expected': bdry})

    # rectangle at the left edge
    mask = [[True, True, True, False],
            [True, True, True, False],
            [True, True, True, False],
            [True, True, True, False]]

    bdry = [[True, True, True, False],
            [True, False, True, False],
            [True, False, True, False],
            [True, True, True, False]]

    output.append({'mask_matrix': mask, 'expected': bdry})

    # rectangle at the right edge
    mask = [[False, True, True, True],
            [False, True, True, True],
            [False, True, True, True],
            [False, True, True, True]]

    bdry = [[False, True, True, True],
            [False, True, False, True],
            [False, True, False, True],
            [False, True, True, True]]

    output.append({'mask_matrix': mask, 'expected': bdry})

    # Rectangle with buffer
    mask = [[False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False]]

    bdry = [[False, False, False, False, False],
            [False, True, True, True, False],
            [False, True, False, True, False],
            [False, True, False, True, False],
            [False, True, True, True, False],
            [False, False, False, False, False]]

    output.append({'mask_matrix': mask, 'expected': bdry})

    # triangle

    mask = [[False, False, False, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False]]

    bdry = [[False, False, False, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, False, True, False, False],
            [True, False, True, True, True],
            [True, True, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False]]

    output.append({'mask_matrix': mask, 'expected': bdry})

    # triangle with hole

    mask = [[False, False, False, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, False, True, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False]]

    bdry = [[False, False, False, False, False],
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, False, True, False, False],
            [True, True, True, True, True],
            [True, True, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False]]

    output.append({'mask_matrix': mask, 'expected': bdry})

    # bizarre
    mask = [[False, True, False, False, False],
            [True, True, True, False, False],
            [False, True, True, True, False],
            [False, True, True, True, False],
            [False, True, False, True, False],
            [False, False, False, True, False],
            [False, False, False, True, False],
            [False, False, True, True, True]]

    bdry = [[False, True, False, False, False],
            [True, False, True, False, False],
            [False, True, False, True, False],
            [False, True, True, True, False],
            [False, True, False, True, False],
            [False, False, False, True, False],
            [False, False, False, True, False],
            [False, False, True, True, True]]

    output.append({'mask_matrix': mask, 'expected': bdry})

    return output


def test_roi_boundary(roi_boundary_data):
    """
    Test that ROI boundary is correctly found
    """
    for pair in roi_boundary_data:
        mask_matrix = pair['mask_matrix']
        expected = pair['expected']
        roi = OphysROI(roi_id=0,
                       x0=0,
                       y0=0,
                       height=len(mask_matrix),
                       width=len(mask_matrix[0]),
                       valid_roi=True,
                       mask_matrix=mask_matrix)

        np.testing.assert_array_equal(roi.boundary_mask,
                                      expected)
