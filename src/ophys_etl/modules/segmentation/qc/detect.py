import numpy as np
from typing import List
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ophys_etl.modules.segmentation.qc_utils import roi_utils
from ophys_etl.modules.segmentation.utils.roi_utils import (
    extract_roi_to_ophys_roi)
from ophys_etl.types import ExtractROI


def roi_metric_qc_plot(
        figure: Figure,
        metric_image: np.ndarray,
        attribute: str,
        roi_list: List[ExtractROI]) -> None:
    """creates a QC plot given some ROIs and a metric image

    Parameters
    ----------
    figure: matplotlib.figure.Figure
        the figure to write into
    metric_image: np.ndarray
        the metric to plot and use for calculating average metric
    attribute: str
        the name of the metric, displayed in a plot title
    roi_list: List[ExtractROI]
        the list of ROIs to plot and from which to compute average metric


    Notes
    -----
    upper left: metric as an image
    upper right: same as upper left, with ROI outlines superimposed
    lower left: scatter plot of n pixels vs average metric per ROI
    lower right: histogram of average metric

    """
    ophys_list = [extract_roi_to_ophys_roi(roi)
                  for roi in roi_list]
    color_map = roi_utils.get_roi_color_map(ophys_list)

    # show the metric image with and without ROIs
    ax00 = figure.add_subplot(2, 2, 1)
    ax00.imshow(metric_image)
    ax01 = figure.add_subplot(2, 2, 2)
    plt_im = ax01.imshow(metric_image, cmap='gray')
    roi_utils.add_rois_to_axes(ax01, roi_list, metric_image.shape,
                               color_map=color_map)
    for ax in [ax00, ax01]:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        figure.colorbar(plt_im, cax=cax)
    ax00.set_title(f"metric: {attribute}")
    ax01.set_title("with ROIs")

    # compute simple measurements of ROIs
    avg_metric_dict = roi_utils.roi_average_metric(roi_list,
                                                   metric_image)
    avg_metric = np.array([avg_metric_dict[i["id"]]
                           for i in roi_list])
    npix = [np.count_nonzero(i["mask"]) for i in roi_list]

    # scatter plot of size vs. avg metric
    ax10 = figure.add_subplot(2, 2, 3)
    ax10.scatter(avg_metric,
                 npix, marker="o",
                 color=plt_im.cmap(avg_metric),
                 edgecolors=None)
    ax10.set_xlabel("average metric per ROI")
    ax10.set_ylabel("n pixels per ROI")

    # histogram of average metric
    ax11 = figure.add_subplot(2, 2, 4)
    if (avg_metric.min() > 0.0) & (avg_metric.max() < 1.0):
        bins = np.linspace(0, 1.0, 50)
    else:
        bins = np.linspace(avg_metric.min(), avg_metric.max(), 50)
    _, _, patches = ax11.hist(avg_metric, bins=bins)
    for patch in patches:
        center = patch.get_bbox().intervalx.mean()
        patch.set_color(plt_im.cmap(center))
    ax11.set_xlabel("average metric per ROI")
    ax11.set_ylabel("n ROIs")
    ax10.set_xlim(ax11.get_xlim())

    figure.tight_layout()
