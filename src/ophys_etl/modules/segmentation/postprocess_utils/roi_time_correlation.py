import numpy as np
import multiprocessing
from ophys_etl.modules.segmentation.postprocess_utils.utils import (
    _winnow_process_list)
from ophys_etl.modules.segmentation.postprocess_utils.roi_types import (
    SegmentationROI)

def _wgts_to_series(sub_video, wgts):
    if len(wgts) == 1:
        return sub_video[:,0]

    # only accept pixels brigter than the median
    th = np.median(wgts)
    wgts -= th
    mask = (wgts<0)
    wgts[mask] = 0.0

    wgts -= wgts.min()
    norm = wgts.max()
    if norm<1.0e-10:
        wgts = np.ones(wgts.shape, dtype=float)
        # re-apply mask on wgts that were
        # originally negative
        wgts[mask] = 0.0
    else:
        wgts = wgts/norm

    if wgts.sum() < 1.0e-10:
        # probably because all weights had
        # the same value, so filtering on the median
        # gave an array of zeros
        wgts = np.ones(wgts.shape, dtype=float)

    n_01 = (wgts>0.01).sum()
    n_001 = (wgts>0.001).sum()
    n_1 = (wgts>0.1).sum()
    print('scaled wgts %.2e %.2e %.2e %d -- %d %d %d'
          % (np.median(wgts), np.quantile(wgts, 0.75), wgts.max(),
             len(wgts),n_1,n_01,n_001))

    key_pixel = np.dot(sub_video, wgts)
    return key_pixel/wgts.sum()

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


def _self_correlate(sub_video, i_pixel):
    npix = sub_video.shape[1]
    ntime = sub_video.shape[0]
    th = np.quantile(sub_video[:,i_pixel], 0.8)
    mask = (sub_video[:,i_pixel]>=th)
    sub_video = sub_video[mask, :]
    this_pixel = sub_video[:, i_pixel]

    this_mu = np.mean(this_pixel)
    other_mu = np.mean(sub_video, axis=0)
    this_var = np.var(this_pixel, ddof=1)
    other_var = np.var(sub_video, axis=0, ddof=1)

    d_other = sub_video-other_mu
    numerator = np.dot((this_pixel-this_mu), d_other)/ntime
    denom = np.sqrt(other_var*this_var)
    corr = numerator/denom
    return np.median(corr)


def _correlate_batch(pixel_list, sub_video, output_dict):
    for ipix in pixel_list:
        value = _self_correlate(sub_video, ipix)
        output_dict[ipix] = value


def get_brightest_pixel_parallel(
        roi: SegmentationROI,
        sub_video: np.ndarray,
        n_processors: int = 8) -> np.ndarray:
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
    npix = sub_video.shape[1]
    chunksize = max(1, npix//(n_processors-1))
    mgr = multiprocessing.Manager()
    output_dict = mgr.dict()
    pix_list = list(range(npix))
    process_list = []
    for i0 in range(0, npix, chunksize):
        chunk = pix_list[i0:i0+chunksize]
        args = (chunk, sub_video, output_dict)
        p = multiprocessing.Process(target=_correlate_batch,
                                    args=args)
        p.start()
        process_list.append(p)
        while len(process_list)>0 and len(process_list)>=(n_processors-1):
            process_list = _winnow_process_list(process_list)
    for p in process_list:
        p.join()

    wgts = np.zeros(npix, dtype=float)
    for ipix in range(npix):
        wgts[ipix] = output_dict[ipix]

    return _wgts_to_series(sub_video, wgts)



def get_brightest_pixel(roi: SegmentationROI,
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
    npix = sub_video.shape[1]
    ntime = sub_video.shape[0]
    wgts = np.zeros(npix, dtype=float)
    for ipix in range(npix):
        wgts[ipix] = _self_correlate(sub_video, ipix)

    return _wgts_to_series(sub_video, wgts)


def calculate_merger_metric(roi0: SegmentationROI,
                            roi1: SegmentationROI,
                            video_lookup: dict,
                            pixel_lookup: dict,
                            self_corr_lookup: dict,
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
    if roi0.area < 2:
        return -999.0

    roi1_video = video_lookup[roi1.roi_id]

    roi0_centroid = pixel_lookup[roi0.roi_id]['key_pixel']

    (roi0_mu,
     roi0_std) = self_corr_lookup[roi0.roi_id]

    roi1_to_roi0 = correlate_sub_video(roi1_video,
                                       roi0_centroid,
                                       filter_fraction=filter_fraction)

    z_score = (roi1_to_roi0-roi0_mu)/roi0_std
    metric = np.median(z_score)
    return metric


def get_self_correlation(sub_video, key_pixel, filter_fraction):
    if sub_video.shape[1] < 2:
        return (0.0, 1.0)

    corr = correlate_sub_video(sub_video,
                               key_pixel,
                               filter_fraction=filter_fraction)

    return (np.mean(corr), np.std(corr, ddof=1))
