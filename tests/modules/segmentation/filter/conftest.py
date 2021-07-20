import pytest
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysROI)


@pytest.fixture
def roi_dict():
    """
    Create a dict of ROIs with IDs [1, 6)
    whose areas = 2*roi_id.

    The dict is keyed on roi_id.
    """

    rng = np.random.default_rng(66523)
    roi_dict = dict()

    for roi_id in range(1, 7, 1):
        x0 = rng.integers(0, 100)
        y0 = rng.integers(0, 100)
        width = rng.integers(10, 11)
        height = rng.integers(10, 15)
        mask = np.zeros(width*height).astype(bool)
        dexes = np.arange(len(mask), dtype=int)
        chosen = rng.choice(dexes, size=2*roi_id, replace=False)
        mask[chosen] = True
        roi = OphysROI(x0=int(x0), width=int(width),
                       y0=int(y0), height=int(height),
                       mask_matrix=mask.reshape((height, width)),
                       roi_id=roi_id,
                       valid_roi=True)
        assert roi.area == 2*roi_id
        roi_dict[roi_id] = roi
    return roi_dict
