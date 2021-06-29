import pytest
from itertools import combinations
import numpy as np

from ophys_etl.modules.segmentation.merge.self_correlation import (
    update_self_correlation)

from ophys_etl.modules.segmentation.merge.roi_time_correlation import (
    get_self_correlation)


@pytest.fixture
def dataset():
    rng = np.random.RandomState(11818)
    video = {}
    key_pixel = {}
    for ii in range(10):
        area = rng.randint(10, 20)
        video[ii] = rng.random_sample((100, area))
        key_pixel[ii] = {'key_pixel': rng.random_sample(100)}
    return {'video': video,
            'key_pixel': key_pixel}


@pytest.mark.parametrize('filter_fraction, n_processors',
                         [(0.2, 2), (0.2, 3),
                          (0.3, 2), (0.3, 3)])
def test_update_self_correlation(dataset, filter_fraction, n_processors):
    merger_candidates = list(combinations(range(10), 2))
    result = update_self_correlation(merger_candidates,
                                     dataset['video'],
                                     dataset['key_pixel'],
                                     filter_fraction,
                                     n_processors)

    assert len(result) == len(dataset['video'])
    for ii in dataset['video']:
        expected = get_self_correlation(
                        dataset['video'][ii],
                        dataset['key_pixel'][ii]['key_pixel'],
                        filter_fraction)
        assert np.abs((expected[0]-result[ii][0])/expected[0]) < 1.0e-6
        assert np.abs((expected[1]-result[ii][1])/expected[1]) < 1.0e-6
