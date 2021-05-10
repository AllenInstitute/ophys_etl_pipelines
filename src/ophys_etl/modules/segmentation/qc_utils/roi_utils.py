from typing import List, Tuple, Callable
import numpy as np
from ophys_etl.modules.decrosstalk.ophys_plane import (
    OphysROI,
    OphysMovie)


def add_roi_boundaries_to_img(img: np.ndarray,
                              roi_list: List[OphysROI],
                              color: Tuple[int] = (255, 0, 0),
                              alpha: float = 0.25) -> np.ndarray:
    """
    Add colored ROI boundaries to an image

    Parameters
    ----------
    img: np.ndarray
        RGB representation of image

    roi_list: List[OphysROI]
        list of ROIs to add to image

    color: Tuple[int]
        color of ROI border as RGB tuple (default: (255, 0, 0))

    alpha: float
        transparency factor to apply to ROI (default=0.25)

    Returns
    -------
    new_img: np.ndarray
        New image with ROI borders superimposed
    """

    new_img = np.copy(img)
    for roi in roi_list:
        bdry = roi.boundary_mask
        for icol in range(roi.width):
            for irow in range(roi.height):
                if not bdry[irow, icol]:
                    continue
                yy = roi.y0 + irow
                xx = roi.x0 + icol
                for ic in range(3):
                    old_val = np.round(img[yy, xx, ic]*(1.0-alpha)).astype(int)
                    new_img[yy, xx, ic] = old_val
                    new_val = np.round(alpha*color[ic]).astype(int)
                    new_img[yy, xx, ic] += new_val

    new_img = np.where(new_img <= 255, new_img, 255)
    return new_img


def roi_thumbnail(movie: OphysMovie,
                  roi: OphysROI,
                  timestamps: np.ndarray,
                  reducer: Callable = np.mean,
                  slop: int = 20,
                  roi_color: Tuple[int] = (255, 0, 0),
                  alpha=0.5) -> np.ndarray:
    """
    Get the thumbnail of an ROI from an OphysMovie

    Parameters
    ----------
    movie: OphysMovie

    roi: OphysROI

    timestamps: np.ndarray
        The timestamps that you want to select when building
        the thumbnail

    reducer: Callable
        The function that will be used to convert
        OphysMovie.data[timestamps,:,:] into an image.
        Must accept the movie array and the kwargs 'axis'
        (Default: np.mean)

    slop: int
        The number of pixels beyond the ROI to return
        in the thumbnail

    roi_color: Tuple[int]
        color of ROI border as RGB tuple (default: (255, 0, 0))

    alpha: float
        transparency factor to apply to ROI (default=0.5)

    Returns
    -------
    thumbnail: np.ndarray
        An RGB representation of the thumbnail with the ROI
        border drawn around it
    """

    clipped_movie = movie.data[timestamps, :, :]
    x0 = roi.x0-slop//2
    x1 = roi.x0+roi.width+slop//2
    y0 = roi.y0-slop//2
    y1 = roi.y0+roi.height+slop//2

    dx = roi.width+slop
    dy = roi.height+slop
    if x0 < 0:
        x0 = 0
        x1 = x0+dx
    if y0 < 0:
        y0 = 0
        y1 = y0+dy
    if x1 >= movie.data.shape[1]:
        x1 = movie.data.shape[1]-1
        x0 = x1-dx
    if y1 >= movie.data.shape[0]:
        y1 = movie.data.shape[0]
        y0 = y1-dy

    clipped_movie = clipped_movie[:, y0:y1, x0:x1]
    mean_img = reducer(clipped_movie, axis=0)
    del clipped_movie

    thumbnail = np.zeros((mean_img.shape[0], mean_img.shape[1], 3),
                         dtype=int)
    v = mean_img.max()
    for ic in range(3):
        thumbnail[:, :, ic] = np.round(255*(mean_img/v)).astype(int)
    thumbnail = np.where(thumbnail <= 255, thumbnail, 255)

    # need to re-center the ROI
    new_roi = OphysROI(x0=roi.x0-x0,
                       y0=roi.y0-y0,
                       width=roi.width,
                       height=roi.height,
                       mask_matrix=roi.mask_matrix,
                       valid_roi=True,
                       roi_id=-999)

    thumbnail = add_roi_boundaries_to_img(thumbnail,
                                          roi_list=[new_roi],
                                          color=roi_color,
                                          alpha=alpha)

    return thumbnail
