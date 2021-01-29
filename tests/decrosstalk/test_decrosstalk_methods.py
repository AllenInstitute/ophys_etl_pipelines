import numpy as np
import ophys_etl.decrosstalk.decrosstalk as decrosstalk


def test_rolling_median_and_std():

    rng = np.random.RandomState(11123)
    data = rng.random_sample(100)
    mask = np.ones(100, dtype=bool)
    med, std = decrosstalk._centered_rolling_mean(data, mask, window=10)
    for ii in range(100):
        i0 = ii-5
        if i0 < 0:
            i0 = 0
        i1 = i0+10
        if i1 > 100:
            i1 = 100
            i0 = 90
        m = np.mean(data[i0:i1])
        s = np.std(data[i0:i1], ddof=1)
        assert abs(m-med[ii]) < 1.0e-10
        assert abs(s-std[ii]) < 1.0e-10

    mask[0:100:2] = False
    med, std = decrosstalk._centered_rolling_mean(data, mask, window=10)
    for ii in range(100):
        i0 = ii-5
        if i0 < 0:
            i0 = 0
        i1 = i0+10
        if i1 > 100:
            i1 = 100
            i0 = 90
        if i0 % 2 == 0:
            m = np.mean(data[i0+1:i1:2])
            s = np.std(data[i0+1:i1:2], ddof=1)
        else:
            m = np.mean(data[i0:i1:2])
            s = np.std(data[i0:i1:2], ddof=1)

        assert abs(m-med[ii]) < 1.0e-10
        assert abs(s-std[ii]) < 1.0e-10
