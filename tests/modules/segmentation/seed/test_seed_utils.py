import pytest
import itertools

from ophys_etl.modules.segmentation.seed.utils import dilated_coordinates


@pytest.mark.parametrize(
        "pixels, dilation_buffer, expected",
        [
            (
                # no dilation
                {(1, 1)}, 0, {(1, 1)}),
            (
                # dilation of 1 pixel
                {(1, 1)}, 1,
                set(itertools.product([0, 1, 2], repeat=2))),
            (
                # dilation of 2 pixels
                {(3, 3)}, 2,
                set(itertools.product([1, 2, 3, 4, 5], repeat=2))),
            ])
def test_dilated_coordinates(pixels, dilation_buffer, expected):
    dilated = dilated_coordinates(pixels, dilation_buffer)
    assert dilated == expected
