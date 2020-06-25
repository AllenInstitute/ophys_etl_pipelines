from typing import Tuple
from pathlib import Path
import pytest

import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.transforms import roi_transforms


@pytest.fixture
def s2p_stat_fixture(tmp_path: Path, request) -> Tuple[Path,
                                                       np.ndarray,
                                                       Tuple[int, int]]:
    """Fixture that allows parametrized mock suite2p stat.npy files to
    be generated.
    """
    movie_shape = request.param.get("movie_shape", (20, 20))
    masks = request.param.get("masks", [np.random.rand(*movie_shape)
                                        for i in range(10)])

    stats = []
    for dense_mask in masks:
        coo = coo_matrix(dense_mask, shape=movie_shape)
        stats.append({"lam": coo.data, "xpix": coo.col, "ypix": coo.row})

    stat_path = tmp_path / "stat.npy"
    np.save(stat_path, stats)
    return (stat_path, masks, movie_shape)


@pytest.mark.parametrize("s2p_stat_fixture", [
    {"movie_shape": (25, 25)},
], indirect=["s2p_stat_fixture"])
def test_suite2p_rois_to_coo(s2p_stat_fixture):
    stat_path, expected_rois, movie_shape = s2p_stat_fixture

    s2p_stat = np.load(stat_path, allow_pickle=True)
    obt_rois = roi_transforms.suite2p_rois_to_coo(s2p_stat, movie_shape)

    for obt_roi, exp_roi in zip(obt_rois, expected_rois):
        assert np.allclose(obt_roi.todense(), exp_roi)


@pytest.mark.parametrize("mask, expected, absolute_threshold, quantile", [
    # test binarize with quantile
    (np.array([[0.0, 0.5, 1.0],
               [0.0, 2.0, 2.0],
               [2.0, 1.0, 0.5]]),
     np.array([[0, 0, 1],
               [0, 1, 1],
               [1, 1, 0]]),
     None,
     0.2),

    # test binarize with absolute_threshold
    (np.array([[0.0, 0.5, 1.0],
               [0.0, 2.0, 2.0],
               [2.0, 1.0, 0.5]]),
     np.array([[0, 0, 0],
               [0, 1, 1],
               [1, 0, 0]]),
     1.5,
     None),

    # test that setting quantile will be ignored if absolute theshold is set
    (np.array([[0.0, 0.5, 1.0],
               [0.0, 2.0, 2.0],
               [2.0, 1.0, 0.5]]),
     np.array([[0, 0, 0],
               [0, 1, 1],
               [1, 0, 0]]),
     1.5,
     0.2),
])
def test_binarize_roi_mask(mask, expected, absolute_threshold, quantile):
    coo_mask = coo_matrix(mask)

    obtained = roi_transforms.binarize_roi_mask(coo_mask,
                                                absolute_threshold,
                                                quantile)

    assert np.allclose(obtained.todense(), expected)
