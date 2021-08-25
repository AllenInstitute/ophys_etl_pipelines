import pytest
import h5py
import numpy as np
import pathlib
import multiprocessing

from ophys_etl.modules.segmentation.merge.louvain_utils import (
    _correlation_worker,
    _correlate_all_pixels)


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
def test_correlate_all_pixels(tmpdir,
                              filter_fraction,
                              n_processors,
                              n_pixels):
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
