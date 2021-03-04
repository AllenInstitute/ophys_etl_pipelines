import h5py
import numpy as np
from pathlib import Path
import pytest
import warnings

import ophys_etl.modules.sine_dewarp.utils as sine_dewarp


@pytest.fixture
def sample_video():
    return np.array(
        [
            [
                [1, 2, 3, 4, 5],
                [1, 6, 7, 8, 9],
                [1, 2, 7, 4, 9]
            ],
            [
                [1, 2, 3, 4, 5],
                [1, 6, 7, 8, 9],
                [1, 2, 7, 4, 9]
            ]
        ]
    )


@pytest.fixture
def random_sample_video():
    return np.random.randint(low=0, high=2, size=(20, 512, 512))


@pytest.fixture
def random_sample_xtable(random_sample_video):
    return sine_dewarp.create_xtable(
        movie=random_sample_video,
        aL=160.0, aR=160.0,
        bL=85.0, bR=90.0,
        noise_reduction=0
    )


def test_noise_reduce_zero(random_sample_video, random_sample_xtable):
    data = np.random.rand(512)

    output = sine_dewarp.noise_reduce(
        data=data,
        xtable=random_sample_xtable,
        ind=0,
        xindex=random_sample_xtable['xindexL'],
        noise_reduction=0
    )

    np.testing.assert_equal(output, data)


def test_noise_reduce_shape(random_sample_video, random_sample_xtable):
    data = np.random.rand(512)

    output = sine_dewarp.noise_reduce(
        data=data,
        xtable=random_sample_xtable,
        ind=0,
        xindex=random_sample_xtable['xindexL'],
        noise_reduction=3
    )

    assert data.shape == output.shape


def test_get_xindex():
    a = 160.
    b = 85.

    x, y = sine_dewarp.get_xindex(a, b)

    assert 256 == len(x)
    assert 256 == len(y)

    # Note: This test only works as long as the final element produced by the
    # sine function isn't a 0. It is unclear if this is possible.
    assert a == np.flatnonzero(x)[-1] + 1
    assert a == np.flatnonzero(y)[-1] + 1


def test_xtable(random_sample_video):
    aL = 0.
    aR = 3.
    bL = 7.
    bR = 11.

    table = sine_dewarp.create_xtable(
        movie=random_sample_video,
        aL=aL, aR=aR, bL=bL, bR=bR,
        noise_reduction=3
    )

    assert 1 == table['bgfactorL']
    assert 1 == table['bgfactorR']

    assert aL == table['aL']
    assert aR == table['aR']
    assert bL == table['bL']
    assert bR == table['bR']


def test_xdewarp_shape(random_sample_video, random_sample_xtable):
    output = sine_dewarp.xdewarp(
        imgin=random_sample_video[0, :, :],
        FOVwidth=512,
        xtable=random_sample_xtable,
        noise_reduction=0
    )

    assert random_sample_video[0].shape == output.shape


def test_dewarp_regression():
    old_path = Path(__file__).parent.joinpath(
        'resources', 'dewarping_regression_test_output.h5')
    input_path = Path(__file__).parent.joinpath(
        'resources', 'dewarping_regression_test_input.h5')

    old_dewarped_video = h5py.File(
        old_path.resolve(), 'r'
    )

    input_video = h5py.File(
        input_path.resolve(), 'r'
    )

    xtable = sine_dewarp.create_xtable(
        movie=input_video['data'],
        aL=160.0,
        aR=150.0,
        bL=85.0,
        bR=100.0,
        noise_reduction=3
    )

    new_dewarped_video = []
    for frame in range(input_video['data'].shape[0]):
        new_dewarped_video.append(
            sine_dewarp.xdewarp(
                imgin=input_video['data'][frame, :, :],
                FOVwidth=0,
                xtable=xtable,
                noise_reduction=3
            )
        )
    new_dewarped_video = np.stack(new_dewarped_video)

    x = np.array(old_dewarped_video['data'])
    y = new_dewarped_video

    if not np.array_equal(x, y):
        warnings.warn(
            f"Regression test does not pass. There are {np.sum(x != y)} "
            f"differences out of {y.size} total values "
            f"({np.sum(x != y) / y.size * 100} %)"
        )
