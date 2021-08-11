import pytest
import numpy as np
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.utils.roi_utils import (
    background_mask_from_roi_list)
from ophys_etl.modules.segmentation.filter.filter_utils import (
    z_vs_background_from_roi)


@pytest.mark.parametrize('clip_quantile', [0.0, 0.2222, 0.3333])
def test_z_vs_background_from_roi(clip_quantile):

    rng = np.random.default_rng(776233)
    img_shape = (32, 32)
    img = rng.normal(1.0, 0.1, img_shape)

    mask = [[True, True, True],
            [True, True, True],
            [True, True, True]]

    roi = OphysROI(x0=10, width=3, y0=10, height=3,
                   valid_roi=True, roi_id=0,
                   mask_matrix=mask)

    img[10:13, 10:13] = 1.3

    background = background_mask_from_roi_list([roi], img_shape)

    z_score = z_vs_background_from_roi(
                  roi,
                  img,
                  background,
                  clip_quantile,
                  n_desired_background=100)
    assert np.allclose(z_score, 3, rtol=0.1, atol=0.0)

    # make sure it can handle case where you ask for
    # more pixels than are available
    z_score = z_vs_background_from_roi(
                  roi,
                  img,
                  background,
                  clip_quantile,
                  n_desired_background=10000)
    assert np.allclose(z_score, 3, rtol=0.1, atol=0.0)

    with pytest.raises(RuntimeError, match='These must be equal'):
        z_vs_background_from_roi(
                  roi,
                  img,
                  np.ones((10, 10), dtype=bool),
                  clip_quantile,
                  n_desired_background=100)

    with pytest.raises(RuntimeError, match='\\[0.0, 1.0\\)'):
        z_vs_background_from_roi(
                  roi,
                  img,
                  np.ones((10, 10), dtype=bool),
                  -0.1,
                  n_desired_background=100)

    with pytest.raises(RuntimeError, match='\\[0.0, 1.0\\)'):
        z_vs_background_from_roi(
                  roi,
                  img,
                  np.ones((10, 10), dtype=bool),
                  1.0001,
                  n_desired_background=100)
