from typing import Union

import pytest

import numpy as np
from scipy.sparse import coo_matrix
import matplotlib

from ophys_etl.qc import roi_qc


PlotType = Union[matplotlib.lines.Line2D, matplotlib.image.AxesImage]


def plot_is_correct(plot_ax: matplotlib.axes.Axes,
                    expected_plot_type: PlotType,
                    expected_data: np.ndarray) -> bool:
    ax_children = plot_ax.get_children()

    obt_data = None
    for child in ax_children:
        if isinstance(child, matplotlib.lines.Line2D):
            # line plots get_ydata() returns an np array
            obt_data = child.get_ydata()
        elif isinstance(child, matplotlib.image.AxesImage):
            # imshow plot get_array() returns an np masked array
            obt_data = child.get_array()
            obt_data = np.ma.asarray(obt_data)

        if obt_data is not None:
            return np.allclose(obt_data, expected_data)

    return False


@pytest.mark.parametrize(("weighted_mask, binary_mask, "
                          "weighted_trace, binary_trace"), [
    (np.array([[0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 2.0, 0.0],
               [0.0, 0.5, 0.2, 0.0],
               [0.0, 0.0, 0.0, 0.0]]),
     np.array([[0, 0, 0, 0],
              [0, 1, 1, 0],
              [0, 1, 0, 0],
              [0, 0, 0, 0]]),
     np.array([0.1, 0.2, 0.3, 0.4]),
     np.array([1, 2, 3, 4]))
])
def test_plot_binarized_vs_weighted_roi(weighted_mask, binary_mask,
                                        weighted_trace, binary_trace):
    weighted_coo_mask = coo_matrix(weighted_mask)
    binary_coo_mask = coo_matrix(binary_mask)

    obtained = roi_qc.plot_binarized_vs_weighted_roi(weighted_coo_mask,
                                                     binary_coo_mask,
                                                     weighted_trace,
                                                     binary_trace)

    # The matplotlib figure will always list axes in same order as long as the
    # plotting function creates plots in the same order
    bin_roi_ax, w_roi_ax, bin_trace_ax, w_trace_ax = obtained.axes

    assert plot_is_correct(bin_roi_ax, matplotlib.image.AxesImage, binary_mask)
    assert plot_is_correct(w_roi_ax, matplotlib.image.AxesImage, weighted_mask)
    assert plot_is_correct(bin_trace_ax, matplotlib.lines.Line2D, binary_trace)
    assert plot_is_correct(w_trace_ax, matplotlib.lines.Line2D, weighted_trace)

    matplotlib.pyplot.close(obtained)
