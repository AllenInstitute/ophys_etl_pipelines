import numpy as np
from typing import Tuple
from ophys_etl.modules.segmentation.postprocess_utils.roi_types import (
    SegmentationROI)
from sklearn.decomposition import PCA as sklearn_PCA


def chisq_from_video(sub_video: np.ndarray,
                     fit_video: np.ndarray,
                     n_components=3):
    """
    sub_video is (nt, npix)
    """
    if sub_video.shape[1] < (n_components+1):
        return 0.0
    npix = sub_video.shape[1]
    ntime = sub_video.shape[0]
    sub_video = sub_video.T

    pca = sklearn_PCA(n_components=n_components, copy=True)

    pca.fit(fit_video)
    transformed_video = pca.fit_transform(sub_video)

    assert transformed_video.shape==(sub_video.shape[0], n_components)

    mu = np.mean(transformed_video, axis=0)
    assert mu.shape == (n_components,)
    distances = np.sqrt(((transformed_video-mu)**2).sum(axis=1))
    assert distances.shape == (npix,)
    std = np.std(distances, ddof=1)
    chisq = ((transformed_video-mu)/std)**2
    chisq = chisq.sum()

    return chisq+2.0*n_components*npix*np.log(std)+n_components*npix*np.log(2.0*np.pi)


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

    #npix_uphill = uphill_video.shape[1]
    #npix_downhill = downhill_video.shape[1]
    #npix = npix_uphill+npix_downhill
    #ntime = video_data.shape[0]

    #merger_video = np.zeros((ntime,
    #                         npix_uphill+npix_downhill), dtype=float)
    #merger_video[:, :npix_uphill] = uphill_video
    #merger_video[:, npix_uphill:] = downhill_video

    uphill_corr = correlate_sub_video(uphill_video,
                                      uphill_centroid,
                                      filter_fraction=filter_fraction)

    #downhill_corr = correlate_sub_video(downhill_video,
    #                                    downhill_centroid,
    #                                    filter_fraction=filter_fraction)

    downhill_to_uphill = correlate_sub_video(downhill_video,
                                             uphill_centroid,
                                             filter_fraction=filter_fraction)

    uphill_mu = np.mean(uphill_corr)
    uphill_std = np.std(uphill_corr, ddof=1)
    z_score = (downhill_to_uphill-uphill_mu)/uphill_std
    metric = np.median(z_score)
    return metric>(-1.0*acceptance)



def validate_merger_bic(uphill_roi: SegmentationROI,
                        downhill_roi: SegmentationROI,
                        video_data: np.ndarray,
                        n_components: int = 3):
    (uphill_video,
     uphill_centroid) = sub_video_from_roi(uphill_roi, video_data)

    (downhill_video,
     downhill_centroid) = sub_video_from_roi(downhill_roi, video_data)

    npix_uphill = uphill_video.shape[1]
    npix_downhill = downhill_video.shape[1]
    npix = npix_uphill+npix_downhill
    ntime = video_data.shape[0]

    merger_video = np.zeros((ntime,
                             npix_uphill+npix_downhill), dtype=float)
    merger_video[:, :npix_uphill] = uphill_video
    merger_video[:, npix_uphill:] = downhill_video

    chisq_uphill = chisq_from_video(uphill_video,
                                    uphill_video,
                                    n_components=n_components)

    chisq_downhill = chisq_from_video(downhill_video,
                                      downhill_video,
                                      n_components=n_components)

    chisq_merger = chisq_from_video(merger_video,
                                    merger_video,
                                    n_components=n_components)

    bic_baseline = 4*n_components*np.log(npix) + chisq_uphill + chisq_downhill
    bic_merger = 2*n_components*np.log(npix) + chisq_merger
    d_bic = bic_merger-bic_baseline
    return (d_bic<0.0)
