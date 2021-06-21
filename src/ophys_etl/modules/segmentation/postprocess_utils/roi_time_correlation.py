import numpy as np
from ophys_etl.modules.segmentation.postprocess_utils.roi_types import (
    SegmentationROI)


def sub_video_from_roi(roi: SegmentationROI,
                       video_data: np.ndarray) -> np.ndarray:
    """
    Get a sub-video that is flattened in space corresponding
    to the video data at the ROI

    Parameters
    ----------
    roi: SegmentationROI

    video_data: np.ndarray
        Shape is (ntime, nrows, ncols)

    Returns
    -------
    sub_video: np.ndarray
        Shape is (ntime, npix) where npix is the number
        of pixels marked True in the ROI
    """

    xmin = roi.x0
    ymin = roi.y0
    xmax = roi.x0+roi.width
    ymax = roi.y0+roi.height

    sub_video = video_data[:, ymin:ymax, xmin:xmax]

    mask = roi.mask_matrix
    sub_video = sub_video[:, mask].reshape(video_data.shape[0], -1)
    return sub_video


def correlate_sub_video(sub_video: np.ndarray,
                        key_pixel: np.ndarray,
                        filter_fraction: float = 0.2) -> np.ndarray:
    """
    Correlated all of the pixels in a sub_video against
    a provided time series using only the brightest N
    timesteps

    Parameters
    ----------
    sub_video: np.ndarray
        Shape is (ntime, npixels)

    key_pixel: np.ndarray
        Shape is (ntime,)
        This is the time series against which to correlate
        the pixels in sub_video

    filter_fraction: float
        Keep the brightest filter_fraction timesteps when doing
        the correlation (this is reckoned by the flux values in
        key_pixel)

    Returns
    -------
    corr: np.ndarray
        Shape is (npix,)
        These are the correlation values of the pixels in
        sub_video against key_pixel
    """

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
                        sub_video: np.ndarray) -> np.ndarray:
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
    #v = sub_video.mean(axis=1)
    #assert v.shape == (sub_video.shape[0],)
    #return v
    xmin = roi.x0
    ymin = roi.y0
    xmax = roi.x0+roi.width
    ymax = roi.y0+roi.height
    sub_img = img_data[ymin:ymax, xmin:xmax]
    mask = roi.mask_matrix
    sub_img = sub_img[mask]
    assert len(sub_video.shape) == 2
    assert sub_video.shape[1] == sub_img.shape[0]
    assert len(sub_img.shape) == 1
    brightest_pixel = np.argmax(sub_img)
    return sub_video[:, brightest_pixel]


def _validate_merger_corr(uphill_roi: SegmentationROI,
                         downhill_roi: SegmentationROI,
                         video_lookup: np.ndarray,
                         img_data: np.ndarray,
                         filter_fraction: float = 0.2,
                         acceptance: float = 1.0) -> bool:
    """
    Validate the merger between two ROIs based on time correlation
    information.

    Parameters
    ----------
    uphill_roi: SegmentationROI

    downhill_roi: SegmentationROI

    video_data: np.ndarray
        Shape is (ntime, nrows, ncols)

    img_data: np.ndarray
        Shape is (nrows, ncols)

    filter_fraction: float
        The fraction of time steps to keep when doing
        time correlation (default = 0.2)

    acceptance: float
        The z-score threshold for accepting the merger
        (default = 1.0)

    Returns
    -------
    boolean

    Notes
    -----
    To assess whether a merger should happen, find the
    brightest pixel in uphill_roi (reckoned with img_data).
    Extract that pixel from video_data as a time series.
    Select only the brightest filter_fraction of pixels
    from that time series. Correlate the rest of the pixels
    in uphill_roi against that brightest_pixel timeseries.
    Use those correlations to construct a Gaussian distribution.

    Correlate the pixels in downhill_roi against that
    brightest_pixel from uphill_roi. Convert these correlations
    into a z-score relative to the Gaussian distribution.
    Accept the merger if the median z-score is greater
    than -1*acceptance. Reject it, otherwise.
    """
    uphill_video = video_lookup[uphill_roi.roi_id]
    downhill_video = video_lookup[downhill_roi.roi_id]

    uphill_centroid = get_brightest_pixel(uphill_roi,
                                          img_data,
                                          uphill_video)

    uphill_corr = correlate_sub_video(uphill_video,
                                      uphill_centroid,
                                      filter_fraction=filter_fraction)

    downhill_to_uphill = correlate_sub_video(downhill_video,
                                             uphill_centroid,
                                             filter_fraction=filter_fraction)

    uphill_mu = np.mean(uphill_corr)
    if len(uphill_corr) > 1:
        uphill_std = np.std(uphill_corr, ddof=1)
    else:
        return False, -999.0

    z_score = (downhill_to_uphill-uphill_mu)/uphill_std
    metric = np.median(z_score)
    return metric > (-1.0*acceptance), metric

def validate_merger_corr(uphill_roi: SegmentationROI,
                         downhill_roi: SegmentationROI,
                         video_lookup: dict,
                         img_data: np.ndarray,
                         filter_fraction: float = 0.2,
                         acceptance: float = 1.0) -> bool:

    #if downhill_roi.mask_matrix.sum() > uphill_roi.mask_matrix.sum():
    #    a = downhill_roi
    #    downhill_roi = uphill_roi
    #    uphill_roi = a

    test1 = _validate_merger_corr(uphill_roi,
                                  downhill_roi,
                                  video_lookup,
                                  img_data,
                                  filter_fraction=filter_fraction,
                                  acceptance=acceptance)

    if test1[0]:
        return test1

    test2 = _validate_merger_corr(downhill_roi,
                                  uphill_roi,
                                  video_lookup,
                                  img_data,
                                  filter_fraction=filter_fraction,
                                  acceptance=acceptance)

    return test2
