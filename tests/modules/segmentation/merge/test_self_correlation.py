import pytest
from itertools import combinations
import numpy as np

from ophys_etl.modules.segmentation.merge.self_correlation import (
    create_self_corr_lookup)

from ophys_etl.modules.segmentation.merge.roi_time_correlation import (
    get_self_correlation)


@pytest.mark.parametrize('filter_fraction, n_processors',
                         [(0.2, 2), (0.2, 3),
                          (0.3, 2), (0.3, 3)])
def test_create_self_corr_lookup(timeseries_and_video_dataset,
                                 filter_fraction,
                                 n_processors):
    merger_candidates = list(combinations(range(10), 2))
    result = create_self_corr_lookup(
                 merger_candidates,
                 timeseries_and_video_dataset['video'],
                 timeseries_and_video_dataset['timeseries'],
                 filter_fraction,
                 n_processors)

    assert len(result) == len(timeseries_and_video_dataset['video'])
    for ii in timeseries_and_video_dataset['video']:
        true_timeseries = timeseries_and_video_dataset['timeseries'][ii]
        expected = get_self_correlation(
                    timeseries_and_video_dataset['video'][ii],
                    true_timeseries['timeseries'],
                    filter_fraction)
        assert np.abs((expected[0]-result[ii][0])/expected[0]) < 1.0e-6
        assert np.abs((expected[1]-result[ii][1])/expected[1]) < 1.0e-6
