import pytest
import numpy as np
from scipy.stats import pearsonr

from ophys_etl.modules.segmentation.validation import video_creation
from ophys_etl.types import ExtractROI


@pytest.mark.parametrize(
        "center, r_radius, c_radius, rotation, shape, id",
        [
            ((100,  200), 4.0, 5.0, 0.0, (512, 512), 1),
            ((100,  200), 4.0, 5.0, 0.1, (512, 512), 2),
            ((100,  200), 4.0, 5.0, 0.4, (512, 512), 3),
            ((100,  200), 4.0, 5.0, 1.4, (512, 512), 4),
            ])
def test_create_roi_ellipse(center, r_radius, c_radius, rotation, shape, id):
    roi = video_creation.create_roi_ellipse(center=center,
                                            r_radius=r_radius,
                                            c_radius=c_radius,
                                            rotation=rotation,
                                            shape=shape,
                                            id=id)
    com = np.argwhere(roi['mask']).mean(axis=0)
    com[0] += roi["y"]
    com[1] += roi["x"]
    np.testing.assert_allclose(com, np.array(center))
    assert roi["width"] < 2.0 * max(r_radius, c_radius)
    assert roi["height"] < 2.0 * max(r_radius, c_radius)
    assert roi["id"] == id


@pytest.fixture
def example_roi(request):
    return ExtractROI(**request.param)


@pytest.mark.parametrize("order", [0, 1, 2])
@pytest.mark.parametrize(
        "example_roi",
        [
            {
                "id": 1,
                "x": 120,
                "y": 235,
                "width": 4,
                "height": 4,
                "valid": True,
                "mask": [[False, True, False, False],
                         [False, True, True, False],
                         [True, True, True, False],
                         [True, True, False, False]]},
                ],
        indirect=["example_roi"])
def test_polynomial_weight_mask(example_roi, order):
    weights = video_creation.polynomial_weight_mask(example_roi, order)

    # weights assigned only to elements in roi mask
    for weight, in_roi in zip(weights.flat,
                              np.array(example_roi["mask"]).flat):
        if in_roi:
            assert weight != 0.0
        else:
            assert weight == 0.0

    # weights normalized to 1.0
    assert weights.max() == 1.0

    # 0-order means all weights are 1.0
    if order == 0:
        np.testing.assert_allclose(
                weights,
                np.array(example_roi["mask"]).astype(float))


@pytest.mark.parametrize(
        "trace, target",
        [
            (np.sin(np.arange(1000)), 0.45),
            (np.sin(np.arange(1000)) + 23.0, 0.45),
            (np.random.randn(1000) + 23.0, 0.95),
            (np.random.randn(1000) + 23.0, 0.45),
            (np.random.randn(1000) + 23.0, 0.15),
            (np.random.randn(1000) + 23.0, 0.0),
            ])
def test_correlated_trace(trace, target):
    new_trace = video_creation.correlated_trace(common_trace=trace,
                                                correlation_target=target)
    if target == 0.0:
        np.testing.assert_allclose(new_trace, np.zeros_like(trace))
    else:
        np.testing.assert_allclose(target,
                                   pearsonr(trace, new_trace)[0])


@pytest.mark.parametrize(
        "common_trace, weights",
        [
            (np.sin(np.arange(1000)), np.array([[0.0, 0.8],
                                                [0.7, 0.6]])),
            ])
def test_correlated_traces_from_weights(common_trace, weights):
    new_traces = video_creation.correlated_traces_from_weights(
            common_trace, weights)
    assert new_traces.shape == (common_trace.size, *weights.shape)
    irows, icols = np.meshgrid(np.arange(weights.shape[0]),
                               np.arange(weights.shape[1]))
    for irow, icol in zip(irows.flat, icols.flat):
        if weights[irow, icol] == 0:
            np.testing.assert_allclose(new_traces[:, irow, icol],
                                       np.zeros_like(common_trace))
        else:
            np.testing.assert_allclose(
                    weights[irow, icol],
                    pearsonr(common_trace, new_traces[:, irow, icol])[0])


def test_movie_with_fake_rois():
    """a smoke test.
    the function being tested is a way to make fake movies that has been
    a little useful for debugging and sanity checks.
    """
    shape = (500, 30, 30)
    movie = video_creation.movie_with_fake_rois(
            spacing=7,
            shape=shape,
            correlation_low=0.1,
            correlation_high=0.95,
            r_radius=2.0,
            c_radius=2.0,
            rotation=0.0,
            n_events=6,
            rate=11.0,
            decay_time=0.4)
    assert movie.shape == shape
