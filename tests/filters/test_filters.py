import pytest
import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.filters import filter_by_aspect_ratio


def coos_by_aspect(threshold):
    width = 100
    height = int(width * threshold)
    while (height/width) <= threshold:
        height += 1
    mdense = np.ones((height, width))
    aspect_larger = [coo_matrix(mdense), coo_matrix(mdense.T)]

    height = int(width * threshold)
    while (height/width) >= threshold:
        height -= 1
    mdense = np.ones((height, width))
    aspect_smaller = [coo_matrix(mdense), coo_matrix(mdense.T)]
    return aspect_smaller, aspect_larger


@pytest.mark.parametrize("threshold", [0.1, 0.2, 0.3])
def test_filter_rois_by_aspect_ratio(threshold):
    csmaller, clarger = coos_by_aspect(threshold)
    filtered = filter_by_aspect_ratio(csmaller + clarger, threshold)
    assert len(clarger) == len(filtered)
    for i, j in zip(clarger, filtered):
        np.testing.assert_array_equal(i.toarray(), j.toarray())
