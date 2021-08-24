import numpy as np
from ophys_etl.modules.segmentation.utils.stats_utils import (
    estimate_std_from_interquartile_range)


def test_std():
    """
    Make sure that the axis parameter to estmate_std_from_interquartile_range
    works as expected
    """
    rng = np.random.default_rng(123112)
    array1d = rng.random(200)
    std = estimate_std_from_interquartile_range(array1d)
    q25, q75 = np.quantile(array1d, (0.25, 0.75))
    expected = (q75-q25)/1.34896
    assert np.abs(std-expected) < 1.0e-10

    array2d = rng.random((75, 62))
    std = estimate_std_from_interquartile_range(array2d)
    q25, q75 = np.quantile(array2d.flatten(), (0.25, 0.75))
    expected = (q75-q25)/1.34896
    assert np.abs(std-expected) < 1.0e-10

    std = estimate_std_from_interquartile_range(array2d, axis=1)
    expected = np.zeros(75)
    for ii in range(75):
        q25, q75 = np.quantile(array2d[ii, :], (0.25, 0.75))
        expected[ii] = (q75-q25)/1.34896
    np.testing.assert_allclose(expected, std)

    std = estimate_std_from_interquartile_range(array2d, axis=0)
    expected = np.zeros(62)
    for ii in range(62):
        q25, q75 = np.quantile(array2d[:, ii], (0.25, 0.75))
        expected[ii] = (q75-q25)/1.34896
    np.testing.assert_allclose(expected, std)
