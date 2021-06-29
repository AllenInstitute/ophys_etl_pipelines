import pytest
import numpy as np
from itertools import combinations

from ophys_etl.modules.segmentation.merge.roi_time_correlation import (
        calculate_merger_metric)

from ophys_etl.modules.segmentation.merge.metric import (
    get_merger_metric_from_pairs)


@pytest.mark.parametrize('filter_fraction, n_processors',
                         [(0.2, 3), (0.2, 2),
                          (0.3, 3), (0.3, 2)])
def test_get_merger_metric_from_pairs(
        timeseries_video_corr_dataset,
        filter_fraction,
        n_processors):

    # TODO: this is currently just a smoke test.
    # More care needs to be put into constructing the test data
    # so that it forces the code to exercise all of the if/else
    # blocks in get_merger_metric_from_pairs

    merger_pairs = list(combinations(range(5), 2))
    actual = get_merger_metric_from_pairs(
                 merger_pairs,
                 timeseries_video_corr_dataset['video'],
                 timeseries_video_corr_dataset['timeseries'],
                 timeseries_video_corr_dataset['self_corr'],
                 filter_fraction,
                 n_processors)

    for pair in merger_pairs:
        video0 = timeseries_video_corr_dataset['video'][pair[0]]
        true_timeseries = timeseries_video_corr_dataset['timeseries'][pair[0]]
        ts0 = true_timeseries['timeseries']
        corr0 = timeseries_video_corr_dataset['self_corr'][pair[0]]

        true_timeseries = timeseries_video_corr_dataset['timeseries'][pair[1]]
        video1 = timeseries_video_corr_dataset['video'][pair[1]]
        ts1 = true_timeseries['timeseries']
        corr1 = timeseries_video_corr_dataset['self_corr'][pair[1]]

        metric01 = calculate_merger_metric(corr0,
                                           ts0,
                                           video1,
                                           filter_fraction=filter_fraction)

        metric10 = calculate_merger_metric(corr1,
                                           ts1,
                                           video0,
                                           filter_fraction=filter_fraction)

        if video0.shape[1] < 2 or video0.shape[1] < 0.5*video1.shape[1]:
            if video1.shape[1] < 2:
                expected = -999.0
            else:
                expected = metric10
        elif video1.shape[1] < 2 or video1.shape[1] < 0.5*video0.shape[1]:
            expected = metric01
        else:
            expected = max(metric01, metric10)

        assert np.abs(expected-actual[pair]) < 1.0e-6
