import pytest
import numpy as np
from typing import List

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.qc_utils import roi_utils
from ophys_etl.types import ExtractROI


@pytest.fixture
def roi_list(request):
    rois: List[ExtractROI] = list()
    for full_mask, roi_id in zip(request.param["full_roi_masks"],
                                 request.param["ids"]):
        coords = np.argwhere(full_mask)
        rowmin, colmin = coords.min(axis=0)
        height, width = np.apply_along_axis(func1d=lambda x: x + 1,
                                            axis=0,
                                            arr=coords.ptp(axis=0))
        mask = [[bool(entry) for entry in row[colmin: (colmin + width)]]
                for row in full_mask[rowmin: (rowmin + height)]]
        rois.append(
                ExtractROI(
                    id=roi_id,
                    x=colmin,
                    y=rowmin,
                    width=width,
                    height=height,
                    valid=True,
                    mask=mask))
    return rois


@pytest.mark.parametrize(
        "metric_image, roi_list, expected",
        [
            (
                np.array([[0.0, 1.0, 2.0, 3.0],
                          [1.0, 2.0, 3.0, 0.0],
                          [3.0, 4.0, 4.0, 5.0],
                          [2.0, 3.0, 4.0, 0.0]]),
                {
                    "ids": [0, 1, 2],
                    "full_roi_masks":
                    [
                        [[1, 1, 0, 0],
                         [1, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0]],
                        [[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 1, 0]],
                        [[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [1, 1, 1, 1]],
                        ]},
                    {0: 1.0, 1: 17.0 / 5, 2: 9.0 / 4}),
            ], indirect=["roi_list"])
def test_roi_average_metric(roi_list, metric_image, expected):
    average_metric = roi_utils.roi_average_metric(roi_list=roi_list,
                                                  metric_image=metric_image)
    assert len(average_metric) == len(expected)
    for roi_id, value in expected.items():
        np.testing.assert_allclose(average_metric[roi_id], value)


@pytest.mark.parametrize('alpha', [0.2, 0.3, 0.4])
def test_add_roi_mask_to_img(alpha):
    rng = np.random.default_rng(634212)
    img = 100*np.ones((64, 64, 3), dtype=int)

    height = 7
    width = 12

    mask = rng.integers(0, 2, (height, width)).astype(bool)
    assert mask.sum() > 0
    roi = OphysROI(x0=20, width=width,
                   y0=15, height=height,
                   valid_roi=True, roi_id=0,
                   mask_matrix=mask)

    color = (22, 33, 44)
    img = roi_utils.add_roi_mask_to_img(
                      img,
                      roi,
                      color,
                      alpha)

    for row in range(height):
        for col in range(width):
            for ic in range(3):
                if not mask[row, col]:
                    assert img[15+row, 20+col, ic] == 100
                else:
                    expected = np.round(alpha*color[ic]
                                        + (1.0-alpha)*100).astype(int)
                    assert img[15+row, 20+col, ic] == expected
