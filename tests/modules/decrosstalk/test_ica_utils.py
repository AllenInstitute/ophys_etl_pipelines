import numpy as np
import scipy.stats
import ophys_etl.modules.decrosstalk.ica_utils as ica_utils


def test_whiten_data():
    """
    test the data whitening routine
    """

    # generate a set of random orthogonal vectors
    # as eigen vectors of the dataset

    # generate data that is independent along those
    # dimensions

    tol = 1.0e-10
    n_dimensions = 4
    rng = np.random.RandomState(2358)

    evec = np.zeros((n_dimensions, n_dimensions), dtype=float)
    for ii in range(n_dimensions):
        v = rng.normal(0, 1, size=n_dimensions)
        for jj in range(ii):
            dot_product = np.dot(v, evec[jj, :])
            v -= dot_product*evec[jj, :]
        norm = np.sqrt(np.sum(v**2))
        assert norm > 0.0
        v /= norm
        evec[ii, :] = v

    n_samples = 1000
    eigen_data = np.zeros((n_samples, n_dimensions), dtype=float)
    sigma = np.random.random_sample(n_dimensions)*5.0+1.0
    mean = np.random.random_sample(n_dimensions)*5.0-2.5
    for i_dim in range(n_dimensions):
        eigen_data[:, i_dim] = rng.normal(mean[i_dim],
                                          sigma[i_dim],
                                          size=n_samples)

    data = np.zeros((n_samples, n_dimensions), dtype=float)
    for i_sample in range(n_samples):
        data[i_sample, :] = np.dot(evec, eigen_data[i_sample, :])

    # make sure that data array is not changed by whiten_data
    data0 = np.copy(data)

    (new_data,
     transformation,
     mean) = ica_utils.whiten_data(data)

    np.testing.assert_array_equal(data0, data)

    for i_dim in range(n_dimensions):
        assert np.abs(mean[i_dim]-np.mean(data[:, i_dim])) < tol

    corr_coef = np.corrcoef(new_data.transpose())
    old_corr_coef = np.corrcoef(data.transpose())
    for ii in range(n_dimensions):
        assert np.abs(1.0-corr_coef[ii, ii]) < tol
        for jj in range(n_dimensions):
            if jj == ii:
                continue
            assert np.abs(corr_coef[ii, jj]) < tol
            assert np.abs(old_corr_coef[ii, jj]) > tol

    # check that the transformation returned works as advertized
    for ii in range(n_dimensions):
        data[:, ii] -= mean[ii]
    data_transformed = np.dot(data, transformation)
    np.testing.assert_array_almost_equal(data_transformed,
                                         new_data,
                                         decimal=10)


def test_fix_source_assignment():
    """
    test that ica_utils.fix_source_assignment works as expected
    """

    n_samples = 1000
    rng = np.random.RandomState(88713)

    signal_1 = np.sort(rng.normal(0.0, 1.0, size=n_samples))
    signal_2 = np.cos(np.linspace(0.0, 1.5*np.pi, n_samples))

    data_0 = np.array([signal_1, signal_2])
    data_1 = np.array([signal_1, signal_2])
    new_data, flag = ica_utils.fix_source_assignment(data_0, data_1)
    assert not flag
    np.testing.assert_array_equal(new_data, data_0)

    data_2 = np.array([signal_2, signal_1])
    new_data, flag = ica_utils.fix_source_assignment(data_0, data_2)
    assert flag
    np.testing.assert_array_equal(new_data, data_0)
    assert not np.array_equal(new_data, data_2)

    signal_3 = np.sort(rng.normal(0.0, 1.1, size=n_samples))
    assert not np.array_equal(signal_3, signal_1)
    signal_4 = np.cos(0.1+np.linspace(0.0, 1.6*np.pi, n_samples))
    assert not np.array_equal(signal_4, signal_2)

    data_3 = np.array([signal_3, signal_4])
    new_data, flag = ica_utils.fix_source_assignment(data_0, data_3)
    assert not flag
    np.testing.assert_array_equal(data_3, new_data)

    data_4 = np.array([signal_4, signal_3])
    new_data, flag = ica_utils.fix_source_assignment(data_0, data_4)
    assert flag
    np.testing.assert_array_equal(data_3, new_data)


def test_run_ica():
    """
    Test actually running ICA
    """
    n_time = 2000
    time_array = np.linspace(0, 100.0, n_time)
    #    signal_1 = np.sin(time_array)
    #    signal_2 = np.cos(time_array)

    signal_1 = np.zeros(n_time, dtype=float)
    signal_2 = np.zeros(n_time, dtype=float)

    for center in (10.0, 46.0, 67.1):
        signal_1 += np.exp(-0.5*(time_array-center)**2)
    for center in (22.3, 48.1, 88.0):
        signal_2 += np.exp(-0.5*(time_array-center)**2)

    signal_2 *= 0.1

    data0 = np.zeros((2, n_time), dtype=float)
    data0[0, :] = 0.9*signal_1+0.1*signal_2
    data0[1, :] = 0.25*signal_1+0.75*signal_2

    for seed in (9, 11):
        # seed 9 triggers swapped==True; I want to make sure
        # we test that edge case, since there was originally
        # a bug in the code that caused run_ica to mis-order
        # the output rows in that case

        rng = np.random.RandomState(seed)
        noise_1 = rng.random_sample(n_time)*0.05
        noise_2 = rng.random_sample(n_time)*0.05
        data = np.copy(data0)
        data[0, :] += noise_1
        data[1, :] += noise_2
        (unmixed_signals,
         mixing_matrix,
         converged,
         swapped) = ica_utils.run_ica(data, 1000, seed, verbose=True)

        assert unmixed_signals.shape == data.shape

        new_data = np.dot(mixing_matrix, unmixed_signals)
        np.testing.assert_array_almost_equal(new_data, data, decimal=10)
        if seed == 9:
            assert swapped
        else:
            assert not swapped
            assert converged

        r10, _ = scipy.stats.pearsonr(signal_1, unmixed_signals[0, :])
        r11, _ = scipy.stats.pearsonr(signal_1, unmixed_signals[1, :])
        r20, _ = scipy.stats.pearsonr(signal_2, unmixed_signals[0, :])
        r21, _ = scipy.stats.pearsonr(signal_2, unmixed_signals[1, :])

        # check that first independent component recreates to first signal
        assert r10 > r11
        assert r10 > 0.9
        assert r11 < 0.05
        assert r10 > r20

        # check that second independent component recreates second signal
        assert r21 > r20
        assert r21 > r11
        assert r21 > 0.6
        assert r20 < 0.1
        assert r21 > r11
