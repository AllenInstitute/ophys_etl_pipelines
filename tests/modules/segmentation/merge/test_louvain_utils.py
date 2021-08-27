import pytest
import h5py
import numpy as np
import pathlib
import multiprocessing

from ophys_etl.modules.segmentation.merge.louvain_utils import (
    _correlation_worker,
    _correlate_all_pixels,
    correlate_all_pixels,
    modularity,
    update_merger_history)


@pytest.mark.parametrize(
    "pixel_range, filter_fraction",
    [((5, 11), 0.2),
     ((7, 16), 0.3)]
)
def test_correlation_worker(tmpdir, pixel_range, filter_fraction):
    """
    Test that a single _correlation_worker writes what is expected
    to the scratch file
    """
    tmpdir_path = pathlib.Path(tmpdir)
    tmpfile_path = tmpdir_path / 'test_corr_worker.h5'

    n_pixels = 20
    dataset_name = 'silly'
    with h5py.File(tmpfile_path, 'w') as out_file:
        out_file.create_dataset(dataset_name,
                                data=np.zeros((n_pixels, n_pixels),
                                              dtype=float),
                                dtype=float)

    rng = np.random.default_rng(762332)
    sub_video = rng.random((1000, n_pixels))

    mgr = multiprocessing.Manager()
    lock = mgr.Lock()

    _correlation_worker(
        sub_video,
        filter_fraction,
        pixel_range,
        lock,
        tmpfile_path,
        dataset_name)

    expected = np.zeros((n_pixels, n_pixels), dtype=float)
    for i0 in range(pixel_range[0], pixel_range[1]):
        trace0 = sub_video[:, i0]
        th = np.quantile(trace0, 1.0-filter_fraction)
        time0 = np.where(trace0 >= th)
        for i1 in range(i0+1, sub_video.shape[1]):
            trace1 = sub_video[:, i1]
            th = np.quantile(trace1, 1.0-filter_fraction)
            time1 = np.where(trace1 >= th)
            timestamps = np.unique(np.concatenate([time0, time1]))
            f0 = trace0[timestamps]
            mu0 = np.mean(f0)
            var0 = np.var(f0, ddof=1)
            f1 = trace1[timestamps]
            mu1 = np.mean(f1)
            var1 = np.var(f1, ddof=1)
            num = np.mean((f0-mu0)*(f1-mu1))
            expected[i0, i1] = num/np.sqrt(var1*var0)

    with h5py.File(tmpfile_path, 'r') as in_file:
        actual = in_file[dataset_name][()]
    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.parametrize(
    "filter_fraction, n_processors, n_pixels",
    [(0.1, 2, 50), (0.1, 3, 50), (0.1, 3, 353),
     (0.2, 2, 50), (0.2, 3, 50), (0.1, 3, 353),
     (0.3, 2, 50), (0.3, 3, 50), (0.1, 3, 353)]
)
def test_correlate_all_pixels_private(tmpdir,
                                      filter_fraction,
                                      n_processors,
                                      n_pixels):
    """
    Test that _correlate_all_pixels returns the upper-diagonal
    correlation matrix
    """
    tmpdir_path = pathlib.Path(tmpdir)
    tmpfile_path = tmpdir_path / 'test_corr_all_pixels.h5'

    rng = np.random.default_rng(887213)
    sub_video = rng.random((1000, n_pixels))

    actual = _correlate_all_pixels(sub_video,
                                   filter_fraction,
                                   n_processors,
                                   tmpfile_path)

    expected = np.zeros((n_pixels, n_pixels), dtype=float)
    for i0 in range(n_pixels):
        trace0 = sub_video[:, i0]
        th = np.quantile(trace0, 1.0-filter_fraction)
        time0 = np.where(trace0 >= th)
        for i1 in range(i0+1, n_pixels):
            trace1 = sub_video[:, i1]
            th = np.quantile(trace1, 1.0-filter_fraction)
            time1 = np.where(trace1 >= th)
            timestamps = np.unique(np.concatenate([time0, time1]))
            f0 = trace0[timestamps]
            mu0 = np.mean(f0)
            var0 = np.var(f0, ddof=1)
            f1 = trace1[timestamps]
            mu1 = np.mean(f1)
            var1 = np.var(f1, ddof=1)
            num = np.mean((f0-mu0)*(f1-mu1))
            expected[i0, i1] = num/np.sqrt(var1*var0)

    np.testing.assert_allclose(expected, actual, atol=1.0e-10, rtol=1.0e-10)


@pytest.mark.parametrize(
    "filter_fraction, n_processors, n_pixels",
    [(0.1, 2, 50), (0.1, 3, 50), (0.1, 3, 353),
     (0.2, 2, 50), (0.2, 3, 50), (0.1, 3, 353),
     (0.3, 2, 50), (0.3, 3, 50), (0.1, 3, 353)]
)
def test_correlate_all_pixels_public(tmpdir,
                                     filter_fraction,
                                     n_processors,
                                     n_pixels):
    """
    Test that correlate_all_pixels returns the full
    correlation matrix
    """
    tmpdir_path = pathlib.Path(tmpdir)

    rng = np.random.default_rng(887213)
    sub_video = rng.random((1000, n_pixels))

    actual = correlate_all_pixels(sub_video,
                                  filter_fraction,
                                  n_processors,
                                  tmpdir_path)

    expected = np.zeros((n_pixels, n_pixels), dtype=float)
    for i0 in range(n_pixels):
        trace0 = sub_video[:, i0]
        th = np.quantile(trace0, 1.0-filter_fraction)
        time0 = np.where(trace0 >= th)
        for i1 in range(i0+1, n_pixels):
            trace1 = sub_video[:, i1]
            th = np.quantile(trace1, 1.0-filter_fraction)
            time1 = np.where(trace1 >= th)
            timestamps = np.unique(np.concatenate([time0, time1]))
            f0 = trace0[timestamps]
            mu0 = np.mean(f0)
            var0 = np.var(f0, ddof=1)
            f1 = trace1[timestamps]
            mu1 = np.mean(f1)
            var1 = np.var(f1, ddof=1)
            num = np.mean((f0-mu0)*(f1-mu1))
            expected[i0, i1] = num/np.sqrt(var1*var0)
            expected[i1, i0] = num/np.sqrt(var1*var0)

    np.testing.assert_allclose(expected, actual, atol=1.0e-10, rtol=1.0e-10)

    tmpfile_list = list(tmpdir_path.glob('**/*'))
    assert len(tmpfile_list) == 0


def test_modularity():
    # test against modularity definition in
    # https://en.wikipedia.org/wiki/Louvain_method
    n_pixels = 100
    possible_roi = (1, 2, 3, 4)

    rng = np.random.default_rng(55332211)

    roi_id = rng.choice(possible_roi, size=n_pixels, replace=True)
    corr = np.zeros((n_pixels, n_pixels), dtype=float)
    for i0 in range(n_pixels):
        corr[i0, i0] = 1.0
        for i1 in range(i0+1, n_pixels):
            v = rng.random()
            corr[i0, i1] = v
            corr[i1, i0] = v

    corr0 = np.copy(corr)
    weight_sum = np.sum(corr, axis=1)

    actual = modularity(roi_id,
                        corr,
                        weight_sum)

    # make sure we did not accidentally change the contents
    # of corr
    np.testing.assert_array_equal(corr, corr0)

    expected = 0.0
    total_weights = 0.0
    for i0 in range(n_pixels):
        for i1 in range(i0+1, n_pixels):
            total_weights += corr[i0, i1]


    for i0 in range(n_pixels):
        for i1 in range(i0+1, n_pixels):
            if roi_id[i0] != roi_id[i1]:
                continue
            expected += corr[i0, i1]
            expected -=(weight_sum[i0]*weight_sum[i1])/(2.0*total_weights)
    expected = expected*0.5/total_weights

    assert np.abs(actual-expected) < 1.0e-10
    assert np.abs(expected) >= 1.0e-5


def test_update_merger_history():
    merger_history = {ii: ii for ii in range(5)}

    merger_history = update_merger_history(
                         merger_history,
                         {'absorbed': 1, 'absorber': 5})

    expected = {ii: ii for ii in range(5)}
    expected[1] = 5
    assert expected == merger_history

    merger_history = update_merger_history(
                         merger_history,
                         {'absorbed': 3, 'absorber': 2})

    expected[3] = 2
    assert expected == merger_history

    merger_history = update_merger_history(
                         merger_history,
                         {'absorbed': 5, 'absorber': 4})

    expected = {ii: ii for ii in range(5)}
    expected[1] = 4
    expected[5] = 4
    expected[3] = 2
    assert expected == merger_history
