import pytest
import numpy as np

import ophys_etl.utils.traces as tx


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.zeros(10), 0.0),    # Zeros
        (np.ones(20), 0.0),     # All same, not zero
        (np.array([-1, -1, -1]), 0.0),   # Negatives
        (np.array([]), np.NaN),   # Empty
        (np.array([0, 0, np.NaN, 0.0]), np.NaN),     # Has NaN
        (np.array([1]), 0.0),    # Unit
        (np.array([-1, 2, 3]), 1.4826)    # Typical
    ]
)
def test_robust_std(x, expected):
    np.testing.assert_equal(expected, tx.robust_std(x))


@pytest.mark.parametrize(
    "x, expected",
    [
        (np.array([0, 1, 2, 3, np.NaN]), np.NaN),
        (np.array([4, 10, 4, 5, 4]), 2.5)
    ])
def test_noise_std(x, expected, monkeypatch):
    monkeypatch.setattr(tx, "robust_std", lambda x: x.max())
    monkeypatch.setattr(tx, "medfilt", lambda x, y: x/2)
    np.testing.assert_equal(expected, tx.noise_std(x))
