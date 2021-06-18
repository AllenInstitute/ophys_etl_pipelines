import numpy as np
from typing import Tuple
from ophys_etl.modules.segmentation.postprocess_utils.roi_types import (
    SegmentationROI)


def sub_video_from_roi(roi: SegmentationROI,
                       video_data: np.ndarray) -> Tuple[np.ndarray,
                                                        np.ndarray]:
    """
    Returns sub-video and centroid pixel
    """

    xmin = roi.x0
    ymin = roi.y0
    xmax = roi.x0+roi.width
    ymax = roi.y0+roi.height

    sub_video = video_data[:,ymin:ymax,xmin:xmax]

    mask = roi.mask_matrix
    sub_video = sub_video[:,mask].reshape(video_data.shape[0], -1)
    return sub_video


def correlate_sub_video(sub_video: np.ndarray,
                        key_pixel: np.ndarray,
                        filter_fraction: float = 0.2):
    npix = sub_video.shape[1]
    discard = 1.0-filter_fraction
    th = np.quantile(key_pixel, discard)
    mask = (key_pixel >= th)
    sub_video = sub_video[mask, :]
    key_pixel = key_pixel[mask]
    ntime = len(key_pixel)
    assert key_pixel.shape == (ntime,)
    assert sub_video.shape == (ntime, npix)

    key_mu = np.mean(key_pixel)
    vid_mu = np.mean(sub_video, axis=0)
    assert vid_mu.shape == (npix,)
    key_var = np.mean((key_pixel-key_mu)**2)
    sub_vid_minus_mu = sub_video-vid_mu
    sub_var = np.mean(sub_vid_minus_mu**2, axis=0)
    assert sub_var.shape == (npix,)

    numerator = np.dot((key_pixel-key_mu), sub_vid_minus_mu)/ntime
    assert numerator.shape == (npix,)

    corr = numerator/np.sqrt(sub_var*key_var)
    assert corr.shape == (npix,)
    return corr


def get_brightest_pixel(roi: SegmentationROI,
                        img_data: np.ndarray,
                        video_data: np.ndarray) -> np.ndarray:
    """
    Return the brightest pixel in an ROI (as measured against
    some image) as a time series.

    Parameters
    ----------
    roi: SegmentationROI

    img_data: np.ndarray
        The image used to assess "brightest pixel".
        Shape is (nrows, ncols).

    video_data: np.ndarray
        Shape is (ntime, nrows, ncols)

    Returns
    -------
    brightest_pixel: np.ndarray
        Time series of taken from the video at the
        brightest pixel in the ROI
    """
    xmin = roi.x0
    ymin = roi.y0
    xmax = roi.x0+roi.width
    ymax = roi.y0+roi.height
    sub_img = img_data[ymin:ymax, xmin:xmax]
    sub_video = video_data[:, ymin:ymax, xmin:xmax]
    mask = roi.mask_matrix
    sub_img = sub_img[mask]
    sub_video = sub_video[:, mask]
    assert len(sub_video.shape) == 2
    assert sub_video.shape[1] == sub_img.shape[0]
    assert len(sub_img.shape) == 1
    brightest_pixel = np.argmax(sub_img)
    return sub_video[:, brightest_pixel]


def validate_merger_corr(uphill_roi: SegmentationROI,
                         downhill_roi: SegmentationROI,
                         video_data: np.ndarray,
                         img_data: np.ndarray,
                         filter_fraction: float=0.2,
                         acceptance: float=1.0):

    uphill_video = sub_video_from_roi(uphill_roi, video_data)
    downhill_video = sub_video_from_roi(downhill_roi, video_data)

    uphill_centroid = get_brightest_pixel(uphill_roi,
                                          img_data,
                                          video_data)

    uphill_corr = correlate_sub_video(uphill_video,
                                      uphill_centroid,
                                      filter_fraction=filter_fraction)

    downhill_to_uphill = correlate_sub_video(downhill_video,
                                             uphill_centroid,
                                             filter_fraction=filter_fraction)

    uphill_mu = np.mean(uphill_corr)
    uphill_std = np.std(uphill_corr, ddof=1)
    z_score = (downhill_to_uphill-uphill_mu)/uphill_std
    metric = np.median(z_score)
    return metric>(-1.0*acceptance)
