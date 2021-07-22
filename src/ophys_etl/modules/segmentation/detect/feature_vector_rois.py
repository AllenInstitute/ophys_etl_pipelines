from typing import Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ophys_etl.utils.array_utils import pairwise_distances


def choose_timesteps(
            sub_video: np.ndarray,
            i_seed: int,
            filter_fraction: float,
            rng: Optional[np.random.RandomState] = None,
            pixel_ignore: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Choose the timesteps to use for a given seed when calculating Pearson
    correlation coefficients.

    Parameters
    ----------
    sub_video: np.ndarray
        A subset of a video to be correlated.
        Shape is (n_time, n_pixels)

    i_seed: int
        The index of the point being considered as the seed
        for the ROI.

    filter_fraction: float
        The fraction of brightest timesteps to be used in calculating
        the Pearson correlation between pixels

    rng: Optional[np.random.RandomState]
        A random number generator used to choose pixels which will be
        used to select the brightest filter_fraction of pixels. If None,
        an np.random.RandomState will be instantiated with a hard-coded
        seed (default: None)

    pixel_ignore: Optional[np.ndarray]:
        An 1-D array of booleans marked True at any pixels
        that should be ignored, presumably because they have already been
        selected as ROI pixels. (default: None)

    Returns
    -------
    global_mask: np.ndarray
        Array of timesteps (ints) to be used for calculating correlations
    """
    # fraction of timesteps to discard
    discard = 1.0-filter_fraction

    # start assembling mask in timesteps
    trace = sub_video[:, i_seed]
    thresh = np.quantile(trace, discard)
    global_mask = []
    mask = np.where(trace >= thresh)[0]
    global_mask.append(mask)

    # choose n_seeds other points to populate global_mask
    possible_seeds = []
    for ii in range(sub_video.shape[1]):
        if ii == i_seed:
            continue

        if pixel_ignore is None or not pixel_ignore[i_seed]:
            possible_seeds.append(ii)

    n_seeds = 10
    if rng is None:
        rng = np.random.RandomState(87123)
    chosen = set()
    chosen.add(i_seed)

    if len(possible_seeds) > n_seeds:
        chosen_seeds = rng.choice(possible_seeds,
                                  size=n_seeds, replace=False)
        for ii in chosen_seeds:
            chosen.add(ii)
    else:
        for ii in possible_seeds:
            chosen.add(ii)

    for chosen_pixel in chosen:
        trace = sub_video[:, chosen_pixel]
        thresh = np.quantile(trace, discard)
        mask = np.where(trace >= thresh)[0]
        global_mask.append(mask)
    global_mask = np.unique(np.concatenate(global_mask))
    return global_mask


def calculate_masked_correlations(
            sub_video: np.ndarray,
            global_mask: np.ndarray,
            pixel_ignore: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate and return the Pearson correlation coefficient between
    pixels in a video, using only specific timesteps

    Parameters
    ----------
    sub_video: np.ndarray
        A subset of a video to be correlated.
        Shape is (n_time, n_pixels)

    global_mask: np.ndarray
        The array of timesteps to be used when calculating the
        Pearson correlation coefficient

    Returns
    -------
    pearson: np.ndarray
        A n_pixels by n_pixels array of Pearson correlation coefficients
        between pixels in the movie.
    """

    # apply timestep mask
    traces = sub_video[global_mask, :].astype(float)

    # calculate the Pearson correlation coefficient between pixels
    mu = np.mean(traces, axis=0)
    traces -= mu
    var = np.mean(traces**2, axis=0)

    n_pixels = traces.shape[1]
    pearson = np.ones((n_pixels,
                       n_pixels),
                      dtype=float)

    numerators = np.tensordot(traces, traces, axes=(0, 0))/sub_video.shape[0]

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

    sub_video = sub_video.reshape(sub_video.shape[0], -1)
    sub_video = sub_video[:, np.logical_not(flat_mask)]

    global_mask = choose_timesteps(sub_video,
                                   i_seed,
                                   filter_fraction,
                                   rng=rng,
                                   pixel_ignore=flat_mask)

    pearson = calculate_masked_correlations(sub_video,
                                            global_mask)

    features = normalize_features(pearson)

    return features


def get_background_mask(
        distances: np.ndarray,
        roi_mask: np.ndarray) -> np.ndarray:
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

    Returns
    -------
    background_mask: np.ndarray
        An (n_pixels,) array of booleans marked True
        for all pixels in the fiducial background of hte ROI


    Notes
    ------
    Background pixels are taken to be the 90% of pixels not in the
    ROI that are the most distant from the pixels that are in the ROI.

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
        # select the 90% most distant pixels to be
        # designated background pixels
        t10 = np.quantile(complement_distances, 0.1)
        valid = complement_distances > t10
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
                p = tuple([int(i)
                           for i in np.unravel_index(ii, self.img_shape)])
                self.index_to_pixel.append(p)
                self.pixel_to_index[p] = len(self.index_to_pixel)-1
                self.n_pixels += 1

        if self.n_pixels == 0:
            raise RuntimeError("Tried to create ROI with no valid pixels")

        self.calculate_feature_distances(sub_video,
                                         filter_fraction,
                                         pixel_ignore=pixel_ignore,
                                         rng=rng)

        self.roi_mask = np.zeros(self.n_pixels, dtype=bool)
        self.roi_mask[self.pixel_to_index[self.seed_pt]] = True

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

        self.feature_distances = pairwise_distances(features)

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
        currently in the ROI).

        4) Any pixel whose distance to the ROI (from step (2)) has a z-score
        of less than -2 relative to the distribution of its distances from
        background pixels (from step (3)) is added to the ROI.
        """
        chose_one = False

        # select background pixels
        background_mask = get_background_mask(self.feature_distances,
                                              self.roi_mask)

        # set ROI distance for every pixel
        d_roi = np.median(self.feature_distances[:, self.roi_mask], axis=1)

        n_roi = self.roi_mask.sum()

        # take the median of the n_roi nearest background distances;
        # hopefully this will limit the effect of outliers
        d_bckgd = np.sort(self.feature_distances[:, background_mask], axis=1)
        if n_roi > 20:
            d_bckgd = d_bckgd[:, :n_roi]

        mu_d_bckgd = np.mean(d_bckgd, axis=1)
        std_d_bckgd = np.std(d_bckgd, axis=1, ddof=1)
        z_score = (d_roi-mu_d_bckgd)/std_d_bckgd

        valid = z_score <= -2.0
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
        features = calculate_pca_feature_vectors(
                                    sub_video=sub_video,
                                    pixel_ignore=pixel_ignore)
        return features
