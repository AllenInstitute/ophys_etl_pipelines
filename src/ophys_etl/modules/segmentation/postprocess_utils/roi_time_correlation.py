import numpy as np
from sklearn.decomposition import PCA as sklearn_pca
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
    discard = 1.0-filter_fraction
    th = np.quantile(key_pixel, discard)
    mask = (key_pixel >= th)
    sub_video = sub_video[mask, :]
    key_pixel = key_pixel[mask]
    ntime = len(key_pixel)

    key_mu = np.mean(key_pixel)
    vid_mu = np.mean(sub_video, axis=0)
    key_var = np.mean((key_pixel-key_mu)**2)
    sub_vid_minus_mu = sub_video-vid_mu
    sub_var = np.mean(sub_vid_minus_mu**2, axis=0)

    numerator = np.dot((key_pixel-key_mu), sub_vid_minus_mu)/ntime

    corr = numerator/np.sqrt(sub_var*key_var)
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
    roi_mask = roi.mask_matrix
    ymin = roi.y0
    ymax = roi.y0+roi.height
    xmin = roi.x0
    xmax = roi.x0+roi.width
    img_data = img_data[ymin:ymax, xmin:xmax]
    img_data = img_data[roi_mask].flatten()


    pca = sklearn_pca(n_components=1)
    transformed = pca.fit_transform(sub_video.transpose())
    assert transformed.shape == (sub_video.shape[1], 1)

    norm = np.dot(img_data,transformed[:,])/np.sum(img_data)

    key_pixel = pca.components_[0, :]
    assert key_pixel.shape == (sub_video.shape[0],)
    return norm*key_pixel


def calculate_merger_metric(roi0: SegmentationROI,
                            roi1: SegmentationROI,
                            video_lookup: dict,
                            img_data: np.ndarray,
                            filter_fraction: float = 0.2) -> float:
    """
    Calculate the merger metric between two ROIs

    Parameters
    ----------
    roi0: SegmentationROI

    roi1: SegmentationROI

    video_lookup: dict
        A dict mapping roi_id to sub-videos like those produced
        by sub_video_from_roi

    img_data: np.ndarray
        Shape is (nrows, ncols)

    filter_fraction: float
        The fraction of time steps to keep when doing
        time correlation (default = 0.2)

    Returns
    -------
    metric: float
        The median z-score of the correlation of roi1's
        pixels to the brightest pixel in roi0 relative
        to the distribution of roi0's pixels to the same.

    Note
    ----
    If there are fewer than 2 pixels in roi0, return -999
    """
    if roi0.mask_matrix.sum() < 2:
        return -999.0

    roi0_video = video_lookup[roi0.roi_id]
    roi1_video = video_lookup[roi1.roi_id]

    roi0_centroid = get_brightest_pixel(roi0,
                                        img_data,
                                        roi0_video)

    roi0_corr = correlate_sub_video(roi0_video,
                                    roi0_centroid,
                                    filter_fraction=filter_fraction)

    roi1_to_roi0 = correlate_sub_video(roi1_video,
                                       roi0_centroid,
                                       filter_fraction=filter_fraction)

    roi0_mu = np.mean(roi0_corr)
    roi0_std = np.std(roi0_corr, ddof=1)

    z_score = (roi1_to_roi0-roi0_mu)/roi0_std
    metric = np.median(z_score)
    return metric
