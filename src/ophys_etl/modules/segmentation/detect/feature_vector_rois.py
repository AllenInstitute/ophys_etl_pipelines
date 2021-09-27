from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ophys_etl.utils.array_utils import pairwise_distances
from ophys_etl.modules.segmentation.utils.stats_utils import (
    estimate_std_from_interquartile_range)
from ophys_etl.modules.segmentation.utils.roi_utils import (
    select_contiguous_region)


def calculate_correlations(
            sub_video: np.ndarray) -> np.ndarray:
    """
    Calculate and return the Pearson correlation coefficient between
    pixels in a video, using only specific timesteps

    Parameters
    ----------
    sub_video: np.ndarray
        A subset of a video to be correlated.
        Shape is (n_time, n_pixels)

    Returns
    -------
    pearson: np.ndarray
        A n_pixels by n_pixels array of Pearson correlation coefficients
        between pixels in the movie.
    """

    # calculate the Pearson correlation coefficient between pixels
    mu = np.mean(sub_video, axis=0)
    sub_video = sub_video - mu
    var = np.mean(sub_video**2, axis=0)

    n_pixels = sub_video.shape[1]
    pearson = np.ones((n_pixels,
                       n_pixels),
                      dtype=float)

    numerators = np.tensordot(sub_video,
                              sub_video, axes=(0, 0))/sub_video.shape[0]

    for ii in range(n_pixels):
        local_numerators = numerators[ii, ii+1:]
        denominators = np.sqrt(var[ii]*var[ii+1:])
        p = local_numerators/denominators
        pearson[ii, ii+1:] = p
        pearson[ii+1:, ii] = p

    return pearson


def normalize_features(input_features: np.ndarray) -> np.ndarray:
    """
    Take an array of feature vectors. Subtract the minimum from
    each feature and normalize by the interquartile range.
    Return the new feature vector.

    Parameters
    ----------
    input_features: np.ndarray

    Returns
    -------
    output_features = np.ndarray
    """
    n_pixels = input_features.shape[0]
    output_features = np.copy(input_features)

    # subtract off minimum of each feature
    input_mins = np.min(input_features, axis=0)
    for ii in range(n_pixels):
        output_features[:, ii] = output_features[:, ii]-input_mins[ii]

    # Normalize by the interquartile range of each feature.
    # If the interquartile range is 0, normalize by max-min.
    # If that is also zero, set the norm to 1.0
    p75 = np.quantile(output_features, 0.75, axis=0)
    p25 = np.quantile(output_features, 0.25, axis=0)
    pmax = np.max(output_features, axis=0)
    pmin = np.min(output_features, axis=0)

    feature_norms = p75-p25
    feature_norms = np.where(feature_norms > 1.0e-20,
                             feature_norms,
                             pmax-pmin)

    feature_norms = np.where(feature_norms > 1.0e-20,
                             feature_norms,
                             1.0)

    for ii in range(n_pixels):
        output_features[:, ii] = output_features[:, ii]/feature_norms[ii]

    return output_features


def calculate_pearson_feature_vectors(
            sub_video: np.ndarray,
            seed_pt: Tuple[int, int],
            pixel_ignore: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate the Pearson correlation-based feature vectors relating
    a grid of pixels in a video to each other

    Parameters
    ----------
    sub_video: np.ndarray
        A subset of a video to be correlated.
        Shape is (n_time, n_rows, n_cols)

        Note: any filtering on time that should be done on
        sub_video must already have been done.

    seed_pt: Tuple[int, int]
        The coordinates of the point being considered as the seed
        for the ROI. Coordinates must be in the frame of the
        sub-video represented by sub_video (i.e. seed_pt=(0,0) will be the
        upper left corner of whatever frame is represented by sub_video)

    pixel_ignore: Optional[np.ndarray]:
        An (n_rows, n_cols) array of booleans marked True at any pixels
        that should be ignored, presumably because they have already been
        selected as ROI pixels. (default: None)

    Returns
    -------
    features: np.ndarray
         An (n_rows*n_cols, n_rows*n_cols) array of feature vectors.
         features[ii, :] is the feature vector for the ii_th pixel
         in sub_video (where ii can be mapped to (row, col) coordinates
         using np.unravel_index()

    Notes
    -----
    The feature vectors returned by this method are calculated as follows:

    1) Calculate the Pearson correlation coefficient between each
    pair of pixels. This results in an (n_pixels, n_pixels) array in which
    pearson[ii, jj] is the correlation coefficient between the ii_th and
    jj_th pixels.

    2) For each column pearson [:, ii] in the array of Pearson
    correlation coefficients, subtract off the minimum value in
    that column and divide by the gap between the 25th adn 75th
    percentile of that column's values.

    Each row in the matrix of Pearson correlation coefficients is now
    a feature vector corresponding to that pixel in sub_video.
    """

    if pixel_ignore is not None:
        flat_mask = pixel_ignore.flatten()
    else:
        flat_mask = np.zeros(sub_video.shape[1]*sub_video.shape[2],
                             dtype=bool)

    # map unmasked i_seed to masked i_seed
    i_seed_0 = np.ravel_multi_index(seed_pt, sub_video.shape[1:])
    i_seed = 0
    for ii in range(len(flat_mask)):
        if ii == i_seed_0:
            break
        if not flat_mask[ii]:
            i_seed += 1

    local_video = sub_video.reshape(sub_video.shape[0], -1)
    local_video = local_video[:, np.logical_not(flat_mask)]
    pearson = calculate_correlations(local_video)
    features = normalize_features(pearson)

    return features


def get_background_mask(
        distances: np.ndarray,
        roi_mask: np.ndarray,
        background_z_score: float = 0.8) -> np.ndarray:
    """
    Find a boolean mask of pixels that are in the fiducial
    background of the field of view.

    Parameters
    ----------
    distances: np.ndarray
        An (n_pixels, n_pixels) array encoding the feature
        space distances between pixels in the field of view

    roi_mask: np.ndarray
        An (n_pixels,) array of booleans marked True
        for all pixels in the ROI

    background_z_score: float
        Pixels that that are more distant from the ROI than
        mean(distance)-background_z_score*std(distance) will be
        considered background pixels (default=1.6).

        If distances are Gaussian distributed,
        background_z_score=1.3 will include the 90% most distant
        pixels as background pixels. background_z_score=1.6 will
        include the 95% most distant.

    Returns
    -------
    background_mask: np.ndarray
        An (n_pixels,) array of booleans marked True
        for all pixels in the fiducial background of hte ROI


    Notes
    ------
    If there are fewer than 10 pixels not already in the ROI, just
    set all of these pixels to be background pixels.
    """
    n_pixels = distances.shape[0]
    complement = np.logical_not(roi_mask)
    n_complement = complement.sum()
    complement_dexes = np.arange(n_pixels, dtype=int)[complement]

    complement_distances = distances[complement, :]
    complement_distances = complement_distances[:, roi_mask]

    # set a pixel's distance from the ROI to be its minimum
    # distance from any pixel in the ROI
    # (in the case where there is only one ROI pixel, these
    # next two lines will not trigger)
    if len(complement_distances.shape) > 1:
        complement_distances = complement_distances.min(axis=1)

    background_mask = np.zeros(n_pixels, dtype=bool)

    if n_complement < 10:
        background_mask[complement] = True
    else:
        std = estimate_std_from_interquartile_range(complement_distances)
        mu = np.median(complement_distances)
        threshold = (mu-background_z_score*std)
        valid = complement_distances > threshold
        valid_dexes = complement_dexes[valid]
        background_mask[valid_dexes] = True

    return background_mask


class PotentialROI(object):
    """
    A class to do the work of finding the pixel mask for a single
    ROI.

    Parameters
    ----------
    seed_pt: Tuple[int, int]
        The global image coordinates of the point from which to
        grow the ROI

    origin: Tuple[int, int]
        The global image coordinates of the upper left pixel in
        sub_video. (seed_pt[0]-origin[0], seed_pt[1]-origin[1])
        will give the row, col coordinates of the seed point
        in sub_video.

    sub_video: np.ndarray
        The segment of the video data to search for the ROI.
        Its shape is (n_time, n_rows, n_cols)

        Note: any filtering on time that needs to be done to sub_video
        must already have been done.

    pixel_ignore: Optional[np.ndarray]
        A (n_rows, n_cols) array of booleans marked True for any
        pixels that should be ignored, presumably because they
        have already been included as ROIs in another iteration.
        (default: None)
    """

    def __init__(self,
                 seed_pt: Tuple[int, int],
                 origin: Tuple[int, int],
                 sub_video: np.ndarray,
                 pixel_ignore: Optional[np.ndarray] = None):
        self.origin = origin
        self.seed_pt = (seed_pt[0]-origin[0], seed_pt[1]-origin[1])
        self.img_shape = sub_video.shape[1:]

        self.index_to_pixel = []
        self.pixel_to_index = {}
        if pixel_ignore is not None:
            pixel_ignore_flat = pixel_ignore.flatten()

        # because feature vectors will handle pixels as a 1-D
        # array, skipping over ignored pixels, create mappings
        # between (row, col) coordinates and 1-D pixel index
        self.n_pixels = 0
        for ii in range(self.img_shape[0]*self.img_shape[1]):
            if pixel_ignore is None or not pixel_ignore_flat[ii]:
                p = tuple([int(i)
                           for i in np.unravel_index(ii, self.img_shape)])
                self.index_to_pixel.append(p)
                self.pixel_to_index[p] = len(self.index_to_pixel)-1
                self.n_pixels += 1

        if self.n_pixels == 0:
            raise RuntimeError("Tried to create ROI with no valid pixels")

        self.calculate_feature_distances(sub_video,
                                         pixel_ignore=pixel_ignore)

        self.roi_mask = np.zeros(self.n_pixels, dtype=bool)
        self.roi_mask[self.pixel_to_index[self.seed_pt]] = True

    def get_features(
            self,
            sub_video: np.ndarray,
            pixel_ignore: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the (n_pixels, n_pixels) array of feature vectors

        Parameters
        ----------
        sub_video: np.ndarray
            The subset of video data being scanned for an ROI.
            Shape is (n_time, n_rows, n_cols)

        pixel_ignore: Optional[np.ndarray]
            A (n_rows, n_cols) array of booleans that is True
            for any pixel to be ignored, presumably because it
            has already been assigned to an ROI (default: None)

        Returns
        -------
        features: np.ndarray
            A (n_pixels, n_features) array in which each row is the
            feature vector corresponding to a pixel in sub_video.

        Notes
        ------
        features[ii, :] can be mapped into a pixel in sub_video using
        self.index_to_pixel[ii]
        """
        msg = "PotentialROI does not implement get_features; "
        msg += "must specify a valid sub-class"
        raise NotImplementedError(msg)

    def calculate_feature_distances(
            self,
            sub_video: np.ndarray,
            pixel_ignore: Optional[np.ndarray] = None):
        """
        Set self.feature_distances, an (n_pixel, n_pixel) array
        of distances between pixels in feature space.

        Parameters
        ----------
        sub_video: np.ndarray
            The subset of video data being scanned for an ROI.
            Shape is (n_time, n_rows, n_cols)

        pixel_ignore: Optional[np.ndarray]
            A (n_rows, n_cols) array of booleans that is True
            for any pixel to be ignored, presumably because it
            has already been assigned to an ROI (default: None)

        Notes
        ------
        self.feature_distances[ii, jj] can be mapped into pixel coordinates
        using self.index_to_pixel[ii], self.index_to_pixel[jj]
        """

        features = self.get_features(sub_video,
                                     pixel_ignore=pixel_ignore)

        self.feature_distances = pairwise_distances(features)

    def select_pixels(self,
                      growth_z_score: float = 3.0,
                      background_z_score: float = 1.3) -> bool:
        """
        Run one iteration, looking for pixels to add to self.roi_mask.

        Parameters
        ----------
        growth_z_score: float
            z-score by which a pixel must prefer correlation with
            ROI pixels over correlation with background pixels
            in order to be added to the ROI (default=3.0)

        background_z_score: float
            When finding background pixels during an iteration
            of ROI growth, select pixels whose minimum distance
            from the ROI in feature space is greater than
            mean(dist)-background_z_score*std(dist)
            (default=1.3)

        Returns
        -------
        chose_one: bool
            True if any pixels were added to the ROI,
            False otherwise.

        Notes
        -----
        The algorithm is as follows.

        1) Use self.get_not_roi_mask() to select pixels that are to be
        designated background pixels

        2) For every candidate pixel, designate its feature space distance
        to the ROI to be the median of its feature space distance to every
        pixel in the ROI.

        3) For every candidate pixel, find the n_roi background pixels that
        are closest to it in feature space (n_roi is the number of pixels
        currently in the ROI).

        4) Any pixel whose distance to the ROI (from step (2)) has a z-score
        of less than -1*growth-z_score relative to the distribution of its
        distances from background pixels (from step (3)) is added to the ROI.
        """
        chose_one = False

        # select background pixels
        background_mask = get_background_mask(
                               self.feature_distances,
                               self.roi_mask,
                               background_z_score=background_z_score)

        # set ROI distance for every pixel
        d_roi = np.median(self.feature_distances[:, self.roi_mask], axis=1)

        n_roi = self.roi_mask.sum()

        # take the median of the n_roi nearest background distances;
        # hopefully this will limit the effect of outliers
        d_bckgd = np.sort(self.feature_distances[:, background_mask], axis=1)
        n_bckgd = max(20, n_roi)
        d_bckgd = d_bckgd[:, :n_bckgd]

        mu_d_bckgd = np.mean(d_bckgd, axis=1)
        if len(d_bckgd.shape) > 1 and d_bckgd.shape[1] > 0:
            std_d_bckgd = estimate_std_from_interquartile_range(d_bckgd,
                                                                axis=1)
        else:
            std_d_bckgd = np.std(d_bckgd, axis=1, ddof=1)
        z_score = (d_roi-mu_d_bckgd)/std_d_bckgd

        valid = z_score <= -1.0*growth_z_score
        if valid.sum() > 0:
            self.roi_mask[valid] = True

        n_roi_1 = self.roi_mask.sum()
        if n_roi_1 > n_roi:
            chose_one = True

        return chose_one

    def get_mask(self,
                 growth_z_score,
                 background_z_score) -> np.ndarray:
        """
        Iterate over the pixels, building up the ROI

        Parameters
        ----------
        growth_z_score: float
            z-score by which a pixel must prefer correlation with
            ROI pixels over correlation with background pixels
            in order to be added to the ROI (default=3.0)

        background_z_score: float
            When finding background pixels during an iteration
            of ROI growth, select pixels whose minimum distance
            from the ROI in feature space is greater than
            mean(dist)-background_z_score*std(dist)

        Returns
        -------
        ouput_img: np.ndarray
            A (n_rows, n_cols) np.ndarray of booleans marked
            True for all ROI pixels
        """
        keep_going = True

        # keep going as long as pizels are being added
        # to the ROI
        while keep_going:
            keep_going = self.select_pixels(growth_z_score,
                                            background_z_score)

        output_img = np.zeros(self.img_shape, dtype=bool)
        for i_pixel in range(self.n_pixels):
            if not self.roi_mask[i_pixel]:
                continue
            p = self.index_to_pixel[i_pixel]
            output_img[p[0], p[1]] = True

        output_img = select_contiguous_region(
                            self.seed_pt,
                            output_img)

        return output_img


class PearsonFeatureROI(PotentialROI):
    """
    A sub-class of PotentialROI that uses the features
    calculated by calculate_pearson_feature_vectors
    to find ROIs.

    See docstring for PotentialROI for __init__ call signature.
    """

    def get_features(
            self,
            sub_video: np.ndarray,
            pixel_ignore: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the (n_pixels, n_pixels) array of feature vectors

        Parameters
        ----------
        sub_video: np.ndarray
            The subset of video data being scanned for an ROI.
            Shape is (n_time, n_rows, n_cols)

        pixel_ignore: Optional[np.ndarray]
            A (n_rows, n_cols) array of booleans that is True
            for any pixel to be ignored, presumably because it
            has already been assigned to an ROI (default: None)

        Returns
        -------
        features: np.ndarray
            A (n_pixels, n_features) array in which each row is the
            feature vector corresponding to a pixel in sub_video.

        Notes
        ------
        features[ii, :] can be mapped into a pixel in sub_video using
        self.index_to_pixel[ii]
        """

        features = calculate_pearson_feature_vectors(
                                    sub_video,
                                    self.seed_pt,
                                    pixel_ignore=pixel_ignore)
        return features


def calculate_pca_feature_vectors(
            sub_video: np.ndarray,
            n_components: int = 50,
            scale: bool = True,
            pixel_ignore: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate the PCA feature vectors relating
    a grid of pixels in a video to each other

    Parameters
    ----------
    sub_video: np.ndarray
        A subset of a video to be correlated.
        Shape is (n_time, n_rows, n_cols)
    n_components: int
        how many PCA components to fit
    scale: bool
        whether to apply StandardScaler()
    pixel_ignore: Optional[np.ndarray]:
        An (n_rows, n_cols) array of booleans marked True at any pixels
        that should be ignored, presumably because they have already been
        selected as ROI pixels. (default: None)

    Returns
    -------
    features: np.ndarray
         An (n_pixels, n_components) array of feature vectors.

    Notes
    -----
    -

    """
    nframes = sub_video.shape[0]
    data = sub_video.reshape(nframes, -1)
    pca = PCA(n_components=n_components)
    pca.fit(data)
    features = pca.components_.T
    if pixel_ignore is not None:
        mask = np.logical_not(pixel_ignore.flatten())
    else:
        mask = np.ones((data.shape[-1])).astype(bool)
    features = features[mask]
    if scale:
        features = StandardScaler().fit_transform(features)
    return features


class PCAFeatureROI(PotentialROI):
    """
    A sub-class of PotentialROI that uses the features
    calculated by calculate_pearson_feature_vectors
    to find ROIs.

    See docstring for PotentialROI for __init__ call signature.
    """

    def get_features(
            self,
            sub_video: np.ndarray,
            pixel_ignore: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Return the (n_pixels, n_pixels) array of feature vectors

        Parameters
        ----------
        sub_video: np.ndarray
            The subset of video data being scanned for an ROI.
            Shape is (n_time, n_rows, n_cols)

        pixel_ignore: Optional[np.ndarray]
            A (n_rows, n_cols) array of booleans that is True
            for any pixel to be ignored, presumably because it
            has already been assigned to an ROI (default: None)

        Returns
        -------
        features: np.ndarray
            A (n_pixels, n_features) array in which each row is the
            feature vector corresponding to a pixel in sub_video.

        Notes
        ------
        features[ii, :] can be mapped into a pixel in sub_video using
        self.index_to_pixel[ii]
        """
        features = calculate_pca_feature_vectors(
                                    sub_video=sub_video,
                                    pixel_ignore=pixel_ignore)
        return features
