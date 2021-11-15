import pytest
import numpy as np
from itertools import product

from ophys_etl.modules.maximum_projection.utils import (
    filter_chunk_of_frames)


@pytest.mark.parametrize('kernel_size', [3, 5, 7])
def test_chunk_of_frames(kernel_size):
    # test will fail on even number kernel sizes;
    # not entirely sure how ndimage handles kernels that don't
    # have a center pixel

    rng = np.random.default_rng(17231)
    chunk_of_frames = rng.random((10, 20, 20))
    assert chunk_of_frames.shape == (10, 20, 20)

    kernel_1d = np.arange(-np.floor(kernel_size//2),
                          kernel_size-np.floor(kernel_size//2),
                          1,
                          dtype=int)

    assert len(kernel_1d) == kernel_size

    expected = np.zeros(chunk_of_frames.shape, dtype=float)
    for i_frame in range(chunk_of_frames.shape[0]):
        for irow in range(chunk_of_frames.shape[1]):
            for icol in range(chunk_of_frames.shape[2]):
                neighbors = np.zeros(kernel_size**2, dtype=float)
                for ineigh, (drow, dcol) in enumerate(product(kernel_1d,
                                                              kernel_1d)):
                    r = irow+drow
                    if r < 0:
                        r = -1*r - 1
                    elif r >= chunk_of_frames.shape[1]:
                        r = 2*(chunk_of_frames.shape[1]-1) - r + 1

                    c = icol+dcol
                    if c < 0:
                        c = -1*c - 1
                    elif c >= chunk_of_frames.shape[2]:
                        c = 2*(chunk_of_frames.shape[2]-1) - c + 1

                    neighbors[ineigh] = chunk_of_frames[i_frame, r, c]
                expected[i_frame, irow, icol] = np.median(neighbors)

    actual = filter_chunk_of_frames(chunk_of_frames, kernel_size)
    np.testing.assert_array_equal(actual, expected)
