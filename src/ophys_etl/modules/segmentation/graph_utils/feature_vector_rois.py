from typing import Optional, Tuple
from scipy.spatial.distance import cdist
import numpy as np
import time


def calculate_pearson_feature_vectors(
            sub_video: np.ndarray,
            seed_pt: Tuple[int, int],
            filter_fraction: float,
            rng: Optional[np.random.RandomState] = None,
            pixel_ignore: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate the Pearson correlation-based feature vectors relating
    a grid of pixels in a video to each other

    Parameters
    ----------
    sub_video: np.ndarray
        A subset of a video to be correlated.
        Shape is (n_time, n_rows, n_cols)

    seed_pt: Tuple[int, int]
        The coordinates of the point being considered as the seed
        for the ROI. Coordinates must be in the frame of the
        sub-video represented by sub_video (i.e. seed_pt=(0,0) will be the
        upper left corner of whatever frame is represented by sub_video)

    filter_fraction: float
        The fraction of brightest timesteps to be used in calculating
        the Pearson correlation between pixels

    rng: Optional[np.random.RandomState]
        A random number generator used to choose pixels which will be
        used to select the brightest filter_fraction of pixels. If None,
        an np.random.RandomState will be instantiated with a hard-coded
        seed (default: None)

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

    1) Select a random set of 10 pixels from sub_video including the seed_pt.

    2) Use the brightest filter_fraction of timestamps from the 10 selected
    seeds to calculate the Pearson correlation coefficient between each
    pair of pixels. This results in an (n_pixels, n_pixels) array in which
    pearson[ii, jj] is the correlation coefficient between the ii_th and
    jj_th pixels.

    3) For each column pearson [:, ii] in the array of Pearson
    correlation coefficients, subtract off the minimum value in
    that column and divide by the gap between the 25th adn 75th
    percentile of that column's values.

    Each row in the matrix of Pearson correlation coefficients is now
    a feature vector corresponding to that pixel in sub_video.
    """

    # fraction of timesteps to discard
    discard = 1.0-filter_fraction

    n_rows = sub_video.shape[1]
    n_cols = sub_video.shape[2]
    #p = np.unravel_index(ii, sub_video.shape[1:])

    # start assembling mask in timesteps

    # apply timestep mask

    if pixel_ignore is not None:
        pixel_ignore_flat = pixel_ignore.flatten()
    else:
        pixel_ignore_flat = np.zeros(n_rows*n_cols, dtype=bool)

    index_to_pixel = []
    for ii in range(len(pixel_ignore_flat)):
        if pixel_ignore_flat[ii]:
            continue
        p = np.unravel_index(ii, sub_video.shape[1:])
        index_to_pixel.append(p)
    n_pixels = len(index_to_pixel)

    pearson = np.ones((n_pixels,
                       n_pixels),
                      dtype=float)

    t0 = time.time()
    for ii0 in range(n_pixels):
        p0 = index_to_pixel[ii0]
        raw_trace0 = sub_video[:, p0[0], p0[1]]
        th0 = np.quantile(raw_trace0, discard)
        mask0 = np.where(raw_trace0>th0)[0]
        for ii1 in range(ii0+1, n_pixels, 1):
            p1 = index_to_pixel[ii1]
            raw_trace1 = sub_video[:, p1[0], p1[1]]
            th1 = np.quantile(raw_trace1, discard)
            mask1 = np.where(raw_trace1>th1)[0]
            mask = np.unique(np.concatenate([mask0, mask1]))
            trace0 = raw_trace0[mask]
            trace1 = raw_trace1[mask]
            mean0 = np.mean(trace0)
            mean1 = np.mean(trace1)
            num = np.mean((trace0-mean0)*(trace1-mean1))
            n = len(mask)
            std0 = np.sum((trace0-mean0)**2)/(n-1)
            std1 = np.sum((trace1-mean1)**2)/(n-1)
            corr = num/np.sqrt(std0*std1)
            pearson[ii0, ii1] = corr
            pearson[ii1, ii0] = corr
        if ii0%100 == 0:
            dur = (time.time()-t0)
            print('pearson ',ii0,dur)

    features = np.copy(pearson)

    # subtract off minimum of each feature
    pearson_mins = np.min(pearson, axis=0)
    for ii in range(n_pixels):
        features[:, ii] = features[:, ii]-pearson_mins[ii]

    # Normalize by the interquartile range of each feature.
    # If the interquartile range is 0, normalize by max-min.
    # If that is also zero, set the norm to 1.0
    p75 = np.quantile(features, 0.75, axis=0)
    p25 = np.quantile(features, 0.25, axis=0)
    pmax = np.max(features, axis=0)
    pmin = np.min(features, axis=0)

    feature_norms = p75-p25
    feature_norms = np.where(feature_norms > 1.0e-20,
                             feature_norms,
                             pmax-pmin)

    feature_norms = np.where(feature_norms > 1.0e-20,
                             feature_norms,
                             1.0)

    for ii in range(n_pixels):
        features[:, ii] = features[:, ii]/feature_norms[ii]

    return features


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

    filter_fraction: float
        The fraction of timesteps to use when correlating pixels
        (this is used by calculate_pearson_feature_vectors)

    pixel_ignore: Optional[np.ndarray]
        A (n_rows, n_cols) array of booleans marked True for any
        pixels that should be ignored, presumably because they
        have already been included as ROIs in another iteration.
        (default: None)

    rng: Optional[np.random.RandomState]
        A random number generator to be used by
        calculate_pearson_feature_vectors in selecting timesteps
        to correlate (default: None)
    """

    def __init__(self,
                 seed_pt: Tuple[int, int],
                 origin: Tuple[int, int],
                 sub_video: np.ndarray,
                 filter_fraction: float,
                 pixel_ignore: Optional[np.ndarray] = None,
                 rng: Optional[np.random.RandomState] = None):
        self.origin = origin
        self.seed_pt = (seed_pt[0]-origin[0], seed_pt[1]-origin[1])
        self.filter_fraction = filter_fraction
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
                p = np.unravel_index(ii, self.img_shape)
                self.index_to_pixel.append(p)
                self.pixel_to_index[p] = len(self.index_to_pixel)-1
                self.n_pixels += 1

        if self.n_pixels == 0:
            raise RuntimeError("Tried to create ROI with no valid pixels")

        self.calculate_feature_distances(sub_video,
                                         filter_fraction,
                                         pixel_ignore=pixel_ignore,
                                         rng=rng)

    def get_features(
            self,
            sub_video: np.ndarray,
            filter_fraction: float,
            pixel_ignore: Optional[np.ndarray] = None,
            rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Return the (n_pixels, n_pixels) array of feature vectors

        Parameters
        ----------
        sub_video: np.ndarray
            The subset of video data being scanned for an ROI.
            Shape is (n_time, n_rows, n_cols)

        filter_fraction: float
            Fraction of brightest timesteps to use in calculating
            features.

        pixel_ignore: Optional[np.ndarray]
            A (n_rows, n_cols) array of booleans that is True
            for any pixel to be ignored, presumably because it
            has already been assigned to an ROI (default: None)

        rng: Optional[np.random.RandomState]
            A random number generator (used by
            calculate_pearson_feature_vectors to select pixels
            for use in choosing brightest timesteps)

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
            filter_fraction: float,
            pixel_ignore: Optional[np.ndarray] = None,
            rng: Optional[np.random.RandomState] = None):
        """
        Set self.feature_distances, an (n_pixel, n_pixel) array
        of distances between pixels in feature space.

        Parameters
        ----------
        sub_video: np.ndarray
            The subset of video data being scanned for an ROI.
            Shape is (n_time, n_rows, n_cols)

        filter_fraction: float
            Fraction of brightest timesteps to use in calculating
            features.

        pixel_ignore: Optional[np.ndarray]
            A (n_rows, n_cols) array of booleans that is True
            for any pixel to be ignored, presumably because it
            has already been assigned to an ROI (default: None)

        rng: Optional[np.random.RandomState]
            A random number generator (used by
            calculate_pearson_feature_vectors to select pixels
            for use in choosing brightest timesteps)

        Notes
        ------
        self.feature_distances[ii, jj] can be mapped into pixel coordinates
        using self.index_to_pixel[ii], self.index_to_pixel[jj]
        """

        features = self.get_features(sub_video,
                                     filter_fraction,
                                     pixel_ignore=pixel_ignore,
                                     rng=rng)

        self.feature_distances = cdist(features,
                                       features,
                                       metric='euclidean')

        self.roi_mask = np.zeros(self.n_pixels, dtype=bool)
        self.roi_mask[self.pixel_to_index[self.seed_pt]] = True

    def get_not_roi_mask(self) -> None:
        """
        Set self.not_roi_mask, a 1-D mask that is marked
        True for pixels that are, for the purposes of one
        iteration, going to be considered part of the background.
        These are taken to be the 90% of pixels not in the ROI that
        are the most distant from the pixels that are in the ROI.

        If there are fewer than 10 pixels not already in the ROI, just
        set all of these pixels to be background pixels.
        """
        complement = np.logical_not(self.roi_mask)
        n_complement = complement.sum()
        complement_dexes = np.arange(self.n_pixels, dtype=int)[complement]

        complement_distances = self.feature_distances[complement, :]
        complement_distances = complement_distances[:, self.roi_mask]

        # set a pixel's distance from the ROI to be its minimum
        # distance from any pixel in the ROI
        # (in the case where there is only one ROI pixel, these
        # next two lines will not trigger)
        if len(complement_distances.shape) > 1:
            complement_distances = complement_distances.min(axis=1)

        self.not_roi_mask = np.zeros(self.n_pixels, dtype=bool)

        if n_complement < 10:
            self.not_roi_mask[complement] = True
        else:
            # select the 90% most distant pixels to be
            # designated background pixels
            t10 = np.quantile(complement_distances, 0.1)
            valid = complement_distances > t10
            valid_dexes = complement_dexes[valid]
            self.not_roi_mask[valid_dexes] = True

        return None

    def select_pixels(self) -> bool:
        """
        Run one iteration, looking for pixels to add to self.roi_mask.

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
        currently in the ROI). The median of these distances is the
        candidate pixel's distance from the background.

        4) Any pixel whose background distance is more than twice its
        ROI distance is added to the ROI
        """
        chose_one = False

        # select background pixels
        self.get_not_roi_mask()

        # set ROI distance for every pixel
        d_roi = np.median(self.feature_distances[:, self.roi_mask], axis=1)

        n_roi = self.roi_mask.sum()

        # take the median of the n_roi nearest background distances;
        # hopefully this will limit the effect of outliers
        d_bckgd = np.sort(self.feature_distances[:, self.not_roi_mask], axis=1)
        d_bckgd = np.median(d_bckgd[:, :n_roi], axis=1)

        valid = d_bckgd > 2*d_roi
        if valid.sum() > 0:
            self.roi_mask[valid] = True

        n_roi_1 = self.roi_mask.sum()
        if n_roi_1 > n_roi:
            chose_one = True

        return chose_one

    def get_mask(self) -> np.ndarray:
        """
        Iterate over the pixels, building up the ROI

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
            keep_going = self.select_pixels()

        output_img = np.zeros(self.img_shape, dtype=bool)
        for i_pixel in range(self.n_pixels):
            if not self.roi_mask[i_pixel]:
                continue
            p = self.index_to_pixel[i_pixel]
            output_img[p[0], p[1]] = True

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
            filter_fraction: float,
            pixel_ignore: Optional[np.ndarray] = None,
            rng: Optional[np.random.RandomState] = None) -> np.ndarray:
        """
        Return the (n_pixels, n_pixels) array of feature vectors

        Parameters
        ----------
        sub_video: np.ndarray
            The subset of video data being scanned for an ROI.
            Shape is (n_time, n_rows, n_cols)

        filter_fraction: float
            Fraction of brightest timesteps to use in calculating
            features.

        pixel_ignore: Optional[np.ndarray]
            A (n_rows, n_cols) array of booleans that is True
            for any pixel to be ignored, presumably because it
            has already been assigned to an ROI (default: None)

        rng: Optional[np.random.RandomState]
            A random number generator (used by
            calculate_pearson_feature_vectors to select pixels
            for use in choosing brightest timesteps)

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
                                    filter_fraction,
                                    pixel_ignore=pixel_ignore,
                                    rng=rng)
        return features
