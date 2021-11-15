import pytest
import numpy as np

from ophys_etl.modules.maximum_projection.utils import (
    reflect_indexes)


@pytest.mark.parametrize(
        'raw_indexes, dim, expected',
        [(np.array([-2, 4, -3, 5, 6, 7]), 6, np.array([2, 4, 3, 5, 4, 3])),
         (np.array([0, 4, 1, 5, 6]), 6, np.array([0, 4, 1, 5, 4])),
         (np.array([-1, 4, 3, 5, 6, 7]), 6, np.array([1, 4, 3, 5, 4, 3])),
         (np.array([-10, 4, -3, 5, 15, 7]), 6, np.array([0, 4, 3, 5, 5, 3]))
         ])
def test_reflect_indexes(raw_indexes, dim, expected):
    reflected_indexes = reflect_indexes(raw_indexes, dim)
    np.testing.assert_array_equal(reflected_indexes, expected)
