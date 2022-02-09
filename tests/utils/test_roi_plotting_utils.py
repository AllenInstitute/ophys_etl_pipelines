import pytest
import numpy as np
from ophys_etl.types import OphysROI
from ophys_etl.utils.roi_plotting_utils import (
    add_roi_contour_to_img)


@pytest.mark.parametrize('alpha', [0.2, 0.3, 0.4])
def test_add_roi_contour_to_img(alpha):
    img = 100*np.ones((64, 64, 3), dtype=int)

    height = 7
    width = 12

    mask = np.zeros((height, width), dtype=bool)
    mask[1, 5:7] = True
    mask[2, 4:8] = True
    mask[3, 3:9] = True
    mask[4, 2:10] = True
    mask[5, 3:9] = True

    bdry_pixels = set([(1, 5), (1, 6), (2, 4), (2, 7),
                       (3, 3), (3, 8), (4, 2), (4, 9),
                       (5, 3), (5, 4), (5, 5), (5, 6),
                       (5, 7), (5, 8)])

    roi = OphysROI(x0=20, width=width,
                   y0=15, height=height,
                   valid_roi=True, roi_id=0,
                   mask_matrix=mask)

    color = (22, 33, 44)
    img = add_roi_contour_to_img(
                      img,
                      roi,
                      color,
                      alpha)

    for row in range(height):
        for col in range(width):
            for ic in range(3):
                if (row, col) not in bdry_pixels:
                    assert img[15+row, 20+col, ic] == 100
                else:
                    expected = np.round(alpha*color[ic]
                                        + (1.0-alpha)*100).astype(int)
                    assert img[15+row, 20+col, ic] == expected
