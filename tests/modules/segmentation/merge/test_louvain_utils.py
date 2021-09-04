import pytest
import h5py
import numpy as np
import pathlib
import multiprocessing

from ophys_etl.utils.array_utils import (
    pairwise_distances)

from ophys_etl.modules.segmentation.utils.roi_utils import (
    pixel_list_to_extract_roi,
    extract_roi_to_ophys_roi)

from ophys_etl.modules.segmentation.merge.louvain_utils import (
    _correlation_worker,
    _correlate_all_pixels,
    correlate_all_pixels,
    modularity,
    update_merger_history,
    _louvain_clustering_iteration,
    _do_louvain_clustering,
    find_roi_clusters)


@pytest.fixture(scope='session')
def rois_for_clustering_fixture():

    pixel_map = [[1, 1, 0, 0, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 2, 2, 0, 0, 0, 0],
                 [0, 0, 2, 2, 0, 0, 5, 0],
                 [0, 0, 0, 0, 0, 0, 5, 0],
                 [0, 0, 0, 0, 0, 0, 5, 0],
                 [3, 3, 0, 0, 0, 0, 0, 0],
                 [3, 3, 0, 4, 4, 4, 4, 0]]

    pixel_map = np.array(pixel_map).astype(int)
    roi_list = []
    for roi_id in range(1, 6):
        valid = np.where(pixel_map==roi_id)
        pixel_list = [(r, c) for r, c in zip(valid[0], valid[1])]
        extract_roi = pixel_list_to_extract_roi(pixel_list, roi_id)
        ophys_roi = extract_roi_to_ophys_roi(extract_roi)
        roi_list.append(ophys_roi)
    return roi_list


@pytest.mark.parametrize(
    'pixel_distance, expected_clusters',
    [(np.sqrt(2.0), set([(1, 2), (3,), (4,), (5,)])),
     (2.0, set([(1, 2), (3, 4, 5)])),
     (3.0, set([(1, 2, 3, 4, 5),]))])
def test_find_roi_clusters(rois_for_clustering_fixture,
                           pixel_distance,
                           expected_clusters):

    clusters = find_roi_clusters(rois_for_clustering_fixture,
                                 pixel_distance=pixel_distance)

    actual = set()
    for roi_cluster in clusters:
        roi_id = [roi.roi_id for roi in roi_cluster]
        roi_id.sort()
        actual.add(tuple(roi_id))
    assert actual == expected_clusters


@pytest.mark.parametrize(
    "pixel_range, filter_fraction, use_pixel_distances, kernel_size",
    [((5, 11), 0.2, False, None),
     ((7, 16), 0.3, False, None),
     ((5, 11), 0.2, True, 1.0),
     ((7, 16), 0.3, True, 1.0),
     ((5, 11), 0.2, True, 5.0),
     ((7, 16), 0.3, True, 5.0)]
)
def test_correlation_worker(tmpdir,
                            pixel_range,
                            filter_fraction,
                            use_pixel_distances,
                            kernel_size):
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

    if use_pixel_distances:
        pixel_distances = np.zeros((n_pixels, n_pixels), dtype=float)
        for i0 in range(n_pixels):
            for i1 in range(i0+1, n_pixels):
                d = rng.random()*7.0
                pixel_distances[i0, i1] = d
                pixel_distances[i1, i0] = d
        too_large = np.where(pixel_distances>kernel_size)
        assert len(too_large[0]) > 0
    else:
        pixel_distances = None

    mgr = multiprocessing.Manager()
    lock = mgr.Lock()

    _correlation_worker(
        sub_video,
        pixel_distances,
        kernel_size,
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
            if kernel_size is not None:
                distance = pixel_distances[i0, i1]
                if distance > kernel_size:
                    continue
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
    "filter_fraction, n_processors, n_pixels, use_pixel_distances, kernel_size",
    [(0.1, 2, 50, False, None),
     (0.1, 3, 50, False, None),
     (0.1, 3, 353, False, None),
     (0.2, 2, 50, False, None),
     (0.2, 3, 50, False, None),
     (0.1, 3, 353, False, None),
     (0.3, 2, 50, False, None),
     (0.3, 3, 50, False, None),
     (0.1, 3, 353, False, None),
     (0.1, 2, 50, True, 1.0),
     (0.1, 3, 50, True, 1.0),
     (0.1, 3, 353, True, 1.0),
     (0.2, 2, 50, True, 1.0),
     (0.2, 3, 50, True, 1.0),
     (0.1, 3, 353, True, 1.0),
     (0.3, 2, 50, True, 1.0),
     (0.3, 3, 50, True, 1.0),
     (0.1, 3, 353, True, 1.0),
     (0.1, 2, 50, True, 5.0),
     (0.1, 3, 50, True, 5.0),
     (0.1, 3, 353, True, 5.0),
     (0.2, 2, 50, True, 5.0),
     (0.2, 3, 50, True, 5.0),
     (0.1, 3, 353, True, 5.0),
     (0.3, 2, 50, True, 5.0),
     (0.3, 3, 50, True, 5.0),
     (0.1, 3, 353, True, 5.0),
     ]
)
def test_correlate_all_pixels_private(tmpdir,
                                      filter_fraction,
                                      n_processors,
                                      n_pixels,
                                      use_pixel_distances,
                                      kernel_size):
    """
    Test that _correlate_all_pixels returns the upper-diagonal
    correlation matrix
    """
    tmpdir_path = pathlib.Path(tmpdir)
    tmpfile_path = tmpdir_path / 'test_corr_all_pixels.h5'

    rng = np.random.default_rng(887213)
    sub_video = rng.random((1000, n_pixels))

    if use_pixel_distances:
        pixel_distances = np.zeros((n_pixels, n_pixels), dtype=float)
        for i0 in range(n_pixels):
            for i1 in range(i0+1, n_pixels):
                d = rng.random()*7.0
                pixel_distances[i0, i1] = d
                pixel_distances[i1, i0] = d
        too_large = np.where(pixel_distances>kernel_size)
        assert len(too_large[0]) > 0
    else:
        pixel_distances = None

    actual = _correlate_all_pixels(sub_video,
                                   pixel_distances,
                                   kernel_size,
                                   filter_fraction,
                                   n_processors,
                                   tmpfile_path)

    expected = np.zeros((n_pixels, n_pixels), dtype=float)
    for i0 in range(n_pixels):
        trace0 = sub_video[:, i0]
        th = np.quantile(trace0, 1.0-filter_fraction)
        time0 = np.where(trace0 >= th)
        for i1 in range(i0+1, n_pixels):
            if kernel_size is not None:
                distance = pixel_distances[i0, i1]
                if distance > kernel_size:
                    continue
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
            expected -= (weight_sum[i0]*weight_sum[i1])/(2.0*total_weights)
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


def test_louvain_clustering_iteration():

    n_pixels = 10
    corr = np.zeros((n_pixels, n_pixels), dtype=float)
    roi_id_arr = np.arange(n_pixels)
    corr[1, 8] = 5.0
    corr[8, 1] = 5.0
    corr[1, 2] = 2.0
    corr[2, 1] = 2.0

    corr0 = np.copy(corr)

    weight_sum_arr = np.sum(corr, axis=1)
    # exclude self correlation
    for ii in range(n_pixels):
        weight_sum_arr -= corr[ii, ii]

    weight0 = np.copy(weight_sum_arr)

    mod0 = modularity(roi_id_arr, corr, weight_sum_arr)

    # best merger is absorb pixel 8 into pixel 1
    (has_changed,
     new_roi_id_arr,
     this_merger) = _louvain_clustering_iteration(
                        roi_id_arr,
                        corr,
                        weight_sum_arr)

    assert has_changed
    assert this_merger == {'absorber': 1, 'absorbed': 8}
    mod1 = modularity(new_roi_id_arr, corr, weight_sum_arr)
    assert mod1 > mod0
    expected = np.array([0, 1, 2, 3, 4, 5, 6, 7, 1, 9])
    np.testing.assert_array_equal(new_roi_id_arr, expected)
    np.testing.assert_array_equal(corr0, corr)
    np.testing.assert_array_equal(weight0, weight_sum_arr)

    # next best merger is to add pixel 2
    (has_changed,
     new_roi_id_arr,
     this_merger) = _louvain_clustering_iteration(
                        new_roi_id_arr,
                        corr,
                        weight_sum_arr)

    assert has_changed
    assert this_merger == {'absorber': 1, 'absorbed': 2}
    mod2 = modularity(new_roi_id_arr, corr, weight_sum_arr)
    assert mod2 > mod1
    expected = np.array([0, 1, 1, 3, 4, 5, 6, 7, 1, 9])
    np.testing.assert_array_equal(new_roi_id_arr, expected)
    np.testing.assert_array_equal(corr0, corr)
    np.testing.assert_array_equal(weight0, weight_sum_arr)

    # should be no more mergers
    roi_input = np.copy(new_roi_id_arr)

    (has_changed,
     new_roi_id_arr,
     this_merger) = _louvain_clustering_iteration(
                        new_roi_id_arr,
                        corr,
                        weight_sum_arr)

    assert not has_changed
    assert this_merger is None
    np.testing.assert_array_equal(roi_input, new_roi_id_arr)


def test_do_louvain_clustering():

    n_pixels = 10
    roi_id_arr = np.arange(n_pixels)
    corr = np.zeros((n_pixels, n_pixels), dtype=float)
    for ii in range(n_pixels):
        corr[ii, ii] = 1.0
    corr[4, 5] = 6.0
    corr[5, 4] = 6.0
    corr[2, 3] = 1.0
    corr[3, 2] = 1.0
    corr[4, 7] = 2.0
    corr[7, 4] = 2.0

    (new_roi_id_arr,
     final_mergers) = _do_louvain_clustering(roi_id_arr,
                                             corr)

    expected = np.array([0, 1, 2, 2, 4, 4, 6, 4, 8, 9])
    np.testing.assert_array_equal(new_roi_id_arr, expected)

    expected_mergers = {ii: ii for ii in range(10)}
    expected_mergers[5] = 4
    expected_mergers[3] = 2
    expected_mergers[7] = 4
    assert expected_mergers == final_mergers
