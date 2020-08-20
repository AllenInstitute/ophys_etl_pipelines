import pytest
import numpy as np
from scipy.sparse import coo_matrix

from ophys_etl.filters import filter_by_aspect_ratio


def coos_by_aspect(threshold):
    width = 100
    height = int(np.ceil(width * threshold)) + 1
    mdense = np.ones((height, width))
    aspect_larger = [coo_matrix(mdense), coo_matrix(mdense.T)]

    height = int(np.floor(width * threshold))
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


def test_filter_rois_by_aspect_ratio_edge_cases():
    threshold = 0.2
    filtered = filter_by_aspect_ratio([], threshold)
    assert filtered == []

    csmaller, clarger = coos_by_aspect(threshold)

    filtered = filter_by_aspect_ratio(csmaller[0:1], threshold)
    assert filtered == []

    filtered = filter_by_aspect_ratio(clarger[0:1], threshold)
    assert filtered == clarger[0:1]
