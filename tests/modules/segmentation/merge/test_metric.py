import pytest
import numpy as np
from itertools import combinations

from ophys_etl.modules.segmentation.merge.roi_time_correlation import (
        calculate_merger_metric)

from ophys_etl.modules.segmentation.merge.metric import (
    get_merger_metric_from_pairs)


@pytest.fixture
def example_data():

    ntime = 50

    rng = np.random.RandomState(712553)
    video_lookup = {}
    key_pixel_lookup = {}
    self_corr_lookup = {}

    area = [1, 10, 2, 6, 5]

    for roi_id, area in enumerate(area):
        area = rng.randint(1,20)
        video_lookup[roi_id] = rng.random_sample((ntime, area))
        key_pixel_lookup[roi_id] = {'key_pixel': rng.random_sample(ntime)}
        self_corr_lookup[roi_id] = (rng.random_sample(),
                                    rng.random_sample())

    return {'video': video_lookup,
            'key_pixel': key_pixel_lookup,
            'self_corr': self_corr_lookup}


@pytest.mark.parametrize('filter_fraction, n_processors',
                         [(0.2, 3), (0.2, 2),
                          (0.3, 3), (0.3, 2)])
def test_get_merger_metric_from_pairs(
        example_data,
        filter_fraction,
        n_processors):

    # TODO: this is currently just a smoke test.
    # More care needs to be put into constructing the test data
    # so that it forces the code to exercise all of the if/else
    # blocks in get_merger_metric_from_pairs

    merger_pairs = list(combinations(range(5), 2))
    actual = get_merger_metric_from_pairs(
                 merger_pairs,
                 example_data['video'],
                 example_data['key_pixel'],
                 example_data['self_corr'],
                 filter_fraction,
                 n_processors)

    for pair in merger_pairs:
        video0 = example_data['video'][pair[0]]
        pix0 = example_data['key_pixel'][pair[0]]['key_pixel']
        corr0 = example_data['self_corr'][pair[0]]

        video1 = example_data['video'][pair[1]]
        pix1 = example_data['key_pixel'][pair[1]]['key_pixel']
        corr1 = example_data['self_corr'][pair[1]]

        metric01 = calculate_merger_metric(corr0,
                                           pix0,
                                           video1,
                                           filter_fraction=filter_fraction)

        metric10 = calculate_merger_metric(corr1,
                                           pix1,
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
