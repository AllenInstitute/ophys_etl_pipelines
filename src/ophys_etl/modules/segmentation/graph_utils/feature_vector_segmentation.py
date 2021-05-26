import matplotlib.pyplot as plt

from typing import Optional, List, Tuple
import networkx as nx
import numpy as np
import multiprocessing
import multiprocessing.managers
from scipy.spatial.distance import cdist
import h5py
import pathlib
import time
import json

from ophys_etl.types import ExtractROI
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

import logging

logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


def graph_to_img(graph_path: pathlib.Path,
                 attribute: str = 'filtered_hnc_Gaussian') -> np.ndarray:
    """
    Convert a graph into a np.ndarray image

    Parameters
    ----------
    graph_path: pathlib.Path
        Path to graph pickle file

    attribute: str
        Name of the attribute used to create the image
        (default = 'filtered_hnc_Gaussian')

    Returns
    -------
    np.ndarray
        An image in which the value of each pixel is the
        sum of the edge weights connected to that node in
        the graph.
    """
    graph = nx.read_gpickle(graph_path)
    node_coords = np.array(graph.nodes).T
    row_max = node_coords[0].max()
    col_max = node_coords[1].max()
    img = np.zeros((row_max+1, col_max+1), dtype=float)
    for node in graph.nodes:
        vals = [graph[node][i][attribute] for i in graph.neighbors(node)]
        img[node[0], node[1]] = np.sum(vals)
    return img


def find_a_peak(img_masked: np.ma.core.MaskedArray,
                mu: float,
                sigma: float,
                n_sigma: int = 2) -> Optional[int]:
    """
    Find a peak in a masked, flattened image array

    Parameters
    ----------
    img_masked: np.ma.core.MaskedArray
        A flattened, masked image array

    mu: float
        The value taken as the mean pixel value for
        assessing peak validity

    sigma: float
        The value taken as the standard deviation of pixel
        values for assessing peak validity

    n_sigma: int
        The number of standard deviations a peak must deviate
        from mu to be considered valid (default: 2)

    Returns
    -------
    i_max: Optional[int]
        The indes of the peak value of img_masked if
        img_masked[i_max] > mu + n_sigma*sigma.
        If not, return None
    """
    candidate = np.argmax(img_masked)
    if img_masked[candidate] > mu+n_sigma*sigma:
        return candidate
    return None


def find_peaks(img: np.ndarray,
               mask: Optional[np.ndarray] = None,
               slop: int = 20,
               n_sigma: int = 2) -> List[dict]:
    """
    Find the peaks in an image to be used as seeds for ROI finding

    Parameters
    ----------
    img: np.ndarray

    mask: Optional[np.ndarray]
        Optional mask set to True in every pixel that has already been
        identified as part of an ROI. These pixels will be ignored in
        peak finding (default: None)

    slop: int
        The number of pixels to each side of a peak that will be masked
        out from further peak consideration on the assumption that they
        are part of the centrail peak's ROI (default: 20)

    n_sigma: int
        The number of sigma a peak must deviate from the median of the
        pixel brightness distribution to be considered an ROI candidate.
        (default: 2)

    Returns
    -------
    seeds: List[dict]
       A list of seeds for ROI finding. Seeds are dicts of the form
       {'center': a tuple of the form (row, col),
        'rows': a tuple of the form (rowmin, rowmax),
        'cols': a tuple of the form (colmin, colmax)}

    Notes
    -----
    This method first calculates mu, the median of all unmasked
    pixel values, and sigma, an estimate of the standard deviation
    of those values taken from the interquartile range. It then
    enters a loop in which it looks for the brightest pixel. If that
    pixel is n_sigma*sigma brighter than mu, that pixel is marked as
    a potential seed. The chosen pixel and a box surrounding it
    (2*slop pixels to a side) are then masked out and the next unmasked
    peak is considered. This process continues until no n_sigma peaks
    are found.
    """

    output = []
    shape = img.shape
    img_flat = img.flatten()

    if mask is None:
        mask_flat = np.zeros(img_flat.shape, dtype=bool)
    else:
        mask_flat = mask.flatten()

    img_masked = np.ma.masked_array(img_flat, mask=mask_flat)
    i25 = np.quantile(img_masked, 0.25)
    i75 = np.quantile(img_masked, 0.75)
    sigma = (i75-i25)/1.349
    mu = np.median(img_masked)

    keep_going = True
    while keep_going:
        candidate = find_a_peak(img_masked,
                                mu,
                                sigma,
                                n_sigma=n_sigma)
        if candidate is None:
            keep_going = False
        else:
            pixel = np.unravel_index(candidate, shape)
            rowmin = max(0, pixel[0]-slop)
            rowmax = min(shape[0], pixel[0]+slop)
            colmin = max(0, pixel[1]-slop)
            colmax = min(shape[1], pixel[1]+slop)

            obj = {'center': (int(pixel[0]), int(pixel[1])),
                   'rows': (int(rowmin), int(rowmax)),
                   'cols': (int(colmin), int(colmax))}
            output.append(obj)
            for irow in range(rowmin, rowmax):
                for icol in range(colmin, colmax):
                    ii = np.ravel_multi_index((irow, icol), shape)
                    img_masked.mask[ii] = True

    return output


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

    # start assembling mask in timesteps
    trace = sub_video[:, seed_pt[0], seed_pt[1]]
    thresh = np.quantile(trace, discard)
    global_mask = []
    mask = np.where(trace >= thresh)[0]
    global_mask.append(mask)

    # choose n_seeds other points to populate global_mask
    i_seed = np.ravel_multi_index(seed_pt, sub_video.shape[1:])
    possible_seeds = []
    for ii in range(n_rows*n_cols):
        if ii == i_seed:
            continue
        p = np.unravel_index(ii, sub_video.shape[1:])

        if pixel_ignore is None or not pixel_ignore[p[0], p[1]]:
            possible_seeds.append(ii)

    n_seeds = 10
    if rng is None:
        rng = np.random.RandomState(87123)
    chosen = set()
    chosen.add(seed_pt)

    if len(possible_seeds) > n_seeds:
        chosen_dex = rng.choice(possible_seeds,
                                size=n_seeds, replace=False)
        for ii in chosen_dex:
             p = np.unravel_index(ii, sub_video.shape[1:])
             chosen.add(p)
    else:
        for ii in possible_seeds:
             p = np.unravel_index(ii, sub_video.shape[1:])
             chosen.add(p)

    for chosen_pixel in chosen:
        trace = sub_video[:, chosen_pixel[0], chosen_pixel[1]]
        thresh = np.quantile(trace, discard)
        mask = np.where(trace >= thresh)[0]
        global_mask.append(mask)
    global_mask = np.unique(np.concatenate(global_mask))

    # apply timestep mask
    sub_video = sub_video[global_mask, :, :]
    shape = sub_video.shape
    n_pixels = shape[1]*shape[2]
    n_time = shape[0]
    traces = sub_video.reshape(n_time, n_pixels).astype(float)
    del sub_video

    # mask pixels that need to be masked
    if pixel_ignore is not None:
        traces = traces[:, np.logical_not(pixel_ignore.flatten())]
        n_pixels = np.logical_not(pixel_ignore).sum()

    # calculate the Pearson correlation coefficient between pixels
    mu = np.mean(traces, axis=0)
    traces -= mu
    var = np.mean(traces**2, axis=0)
    traces = traces.transpose()

    pearson = np.ones((n_pixels,
                       n_pixels),
                      dtype=float)

    numerators = np.tensordot(traces, traces, axes=(1, 1))/n_time

    for ii in range(n_pixels):
        local_numerators = numerators[ii, ii+1:]
        denominators = np.sqrt(var[ii]*var[ii+1:])
        p = local_numerators/denominators
        pearson[ii, ii+1:] = p
        pearson[ii+1:, ii] = p

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
    feature_norms = np.where(feature_norms>1.0e-20,
                             feature_norms,
                             pmax-pmin)

    feature_norms = np.where(feature_norms>1.0e-20,
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


def _get_roi(seed_obj: dict,
             video_data: np.ndarray,
             filter_fraction: float,
             pixel_ignore: np.ndarray,
             output_dict: multiprocessing.managers.DictProxy,
             roi_id: int,
             roi_class: type) -> None:
    """
    Worker method started as a multiprocessing.Process
    to actually grow an ROI.

    Parameters
    ----------
    seed_obj: dict
        A dict as returned by find_peaks containing the seed
        information for one ROI

    video_data: np.ndarray
        The subset of a video to be searched for the ROI. Shape
        is (n_time, n_rows, n_cols) where n_rows and n_cols
        are equal to seed_obj['rows'][1]-seed_obj['rows'][0]
        and seed_boj['cols'][1]-seed_obj['cols'][0] (i.e. the
        field of view has already been clipped)

    filter_fraction: float
        The fraction of brightest timesteps to use in feature
        calculation

    pixel_ignore: np.ndarray
        A (n_rows, n_cols) array of booleans marked True at
        any pixels that should be ignored, presumably because
        they have already been added to an ROI

    output_dict: multiprocessing.managers.DictProxy
        The dict where the final ROI mask from this search will
        be stored. After running this method,
        output_dict[roi_id] will be a tuple containing the
        origin of the field of view (i.e. the coordinates of the
        upper left corner) and the array of booleans representing
        the ROI's mask.

    roi_id: int
        The unique ID of this ROI. This will be used as the key
        in output_dict for this ROI's mask

    roi_class: type
        The sub-class of PotentialROI that will be used to find
        this ROI

    Returns
    -------
    None
        Results are stored in output_dict
    """

    seed_pt = seed_obj['center']
    origin = (seed_obj['rows'][0], seed_obj['cols'][0])

    roi = roi_class(seed_pt,
                    origin,
                    video_data,
                    filter_fraction,
                    pixel_ignore=pixel_ignore)

    final_mask = roi.get_mask()
    output_dict[roi_id] = (origin, final_mask)
    return None


def convert_to_lims_roi(origin: Tuple[int, int],
                        mask: np.ndarray,
                        roi_id: int = 0) -> ExtractROI:
    """
    Convert an origin and a pixel mask into a LIMS-friendly
    JSONized ROI

    Parameters
    ----------
    origin: Tuple[int, int]
        The global coordinates of the upper right corner of the pixel mask

    mask: np.ndarray
        A 2D array of booleans marked as True at the ROI's pixels

    roi_id: int
        default: 0

    Returns
    --------
    roi: dict
        an ExtractROI matching the input data
    """
    # trim mask
    row0 = 0
    col0 = 0
    row1 = mask.shape[0]
    col1 = mask.shape[1]
    for i_row in range(mask.shape[0]):
        if mask[i_row, :].sum() > 0:
            break
        row0 += 1
    for i_row in range(mask.shape[0]-1, -1, -1):
        if mask[i_row, :].sum() > 0:
            break
        row1 -= 1
    for i_col in range(mask.shape[1]):
        if mask[:, i_col].sum() > 0:
            break
        col0 += 1
    for i_col in range(mask.shape[1]-1, -1, -1):
        if mask[:, i_col].sum() > 0:
            break
        col1 -= 1

    new_mask = mask[row0:row1, col0:col1]
    roi = ExtractROI(id=roi_id,
                     x=int(origin[1]+col0),
                     y=int(origin[0]+row0),
                     width=int(col1-col0),
                     height=int(row1-row0),
                     valid=False,
                     mask=[i.tolist() for i in new_mask])
    return roi


def create_roi_plot(plot_path: pathlib.Path,
                    img_data: np.ndarray,
                    roi_list: List[ExtractROI]) -> None:
    """
    Generate a side-by-side plot comparing the image data
    used to seed ROI generation with the borders of the
    discovered ROIs

    Parameters
    ----------
    plot_path: pathlib.Path
        Path to file where plot will be saved

    img_data: np.ndarray
        The baseline image over which to plot the ROIs

    roi_list: List[ExtractROI]

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(40, 20))
    axes[0].imshow(img_data)
    axes[1].imshow(img_data)

    bdry_pixels = np.zeros(img_data.shape, dtype=int)
    for roi in roi_list:
        ophys_roi = OphysROI(
                        roi_id=0,
                        x0=roi['x'],
                        y0=roi['y'],
                        width=roi['width'],
                        height=roi['height'],
                        valid_roi=False,
                        mask_matrix=roi['mask'])

        bdry = ophys_roi.boundary_mask
        for ir in range(ophys_roi.height):
            for ic in range(ophys_roi.width):
                if bdry[ir, ic]:
                    bdry_pixels[ir+ophys_roi.y0,
                                ic+ophys_roi.x0] = 1

    bdry_pixels = np.ma.masked_where(bdry_pixels == 0,
                                     bdry_pixels)
    axes[1].imshow(bdry_pixels, cmap='autumn', alpha=0.5)
    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig=fig)
    return None


class FeatureVectorSegmenter(object):
    """
    A class that looks for ROIs based on the clustering of pixels
    in a feature space calculated from video data.

    Parameters
    ----------
    graph_input: pathlib.Path
        Path to a graph which will be used to seed locations for
        ROIs (ROIs are detected from features that are calculated
        directly from the video data)

    video_input: pathlib.Path
        Path to the video in which ROIs will be detected

    attribute: str
        The name of the edge attribute that will be used to construct
        an image from graph_input. Peaks in that image will be used to
        seed ROIs (default: 'filtered_hnc_Gaussian')

    filter_fraction: float
        The fraction of brightest timesteps that will be used to construct
        features from the video data

    n_processors: int
        The number of parallel processors to use when searching for ROIs
        (default: 8)

    roi_class: type
        The sub-class of PotentialROI that is used to grow ROIs from a seed
        to a mask (default: PearsonFeatureROI)

    Notes
    -----
    After calling the run() method in this class, ROIs will be written to
    a JSONised list. There are also options to store the pixel location of
    seeds used to find ROIs at each iteration, as well as a summary plot
    showing ROI borders superimposed over the image derived from graph_input.
    """

    def __init__(self,
                 graph_input: pathlib.Path,
                 video_input: pathlib.Path,
                 attribute: str = 'filtered_hnc_Gaussian',
                 filter_fraction: float = 0.2,
                 n_processors=8,
                 roi_class=PearsonFeatureROI):

        self.roi_class = roi_class
        self.n_processors = n_processors
        self._attribute = attribute
        self._graph_input = graph_input
        self._video_input = video_input
        self._filter_fraction = filter_fraction
        self.rng = np.random.RandomState(11923141)
        self._graph_img = graph_to_img(graph_input,
                                       attribute=attribute)

        with h5py.File(self._video_input, 'r') as in_file:
            self._movie_data = in_file['data'][()]
            movie_shape = self._movie_data.shape
            if movie_shape[1:] != self._graph_img.shape:
                msg = f'movie shape: {movie_shape}\n'
                msg += f'img shape: {self._graph_img.shape}'
                raise RuntimeError(msg)

    def _run(self,
             img_data: np.ndarray,
             video_data: np.ndarray) -> List[dict]:
        """
        Run one iteration of ROI detection

        Parameters
        ----------
        img_data: np.ndarray
            A (n_rows, n_cols) array representing an image in which
            to detect peaks around which to grow ROIs

        video_data: np.ndarray
            A (n_time, n_rows, n_cols) array containing the video data
            used to detect ROIs

        Returns
        -------
        seed_list: List[dict]
            A list of all of the seeds (as returned by find_peaks)
            investigated during this iteration of ROI finding

        Notes
        -----
        As this method is run, it will add any pixels identified as
        ROI pixels to self.roi_pixels. Individual ROIs will be
        appended to self.roi_list.
        """

        seed_list = find_peaks(img_data,
                               mask=self.roi_pixels,
                               slop=20)

        logger.info(f'got {len(seed_list)} seeds')

        # ROIs can be grown independently of each other;
        # farm each seed out to an independent process
        p_list = []
        mgr = multiprocessing.Manager()
        mgr_dict = mgr.dict()
        for i_seed, seed in enumerate(seed_list):
            self.roi_id += 1
            mask = self.roi_pixels[seed['rows'][0]:seed['rows'][1],
                                   seed['cols'][0]:seed['cols'][1]]

            video_data_subset = video_data[:,
                                           seed['rows'][0]:seed['rows'][1],
                                           seed['cols'][0]:seed['cols'][1]]

            p = multiprocessing.Process(target=_get_roi,
                                        args=(seed,
                                              video_data_subset,
                                              self._filter_fraction,
                                              mask,
                                              mgr_dict,
                                              self.roi_id,
                                              self.roi_class))
            p.start()
            p_list.append(p)

            # make sure that all processors are working at all times,
            # if possible
            while len(p_list) > 0 and len(p_list) >= self.n_processors-1:
                to_pop = []
                for ii in range(len(p_list)-1, -1, -1):
                    if p_list[ii].exitcode is not None:
                        to_pop.append(ii)
                for ii in to_pop:
                    p_list.pop(ii)
        for p in p_list:
            p.join()

        logger.info('all processes complete')

        # write output from individual processes to
        # class storage variables
        for roi_id in mgr_dict:
            origin = mgr_dict[roi_id][0]
            mask = mgr_dict[roi_id][1]
            roi = convert_to_lims_roi(origin,
                                      mask,
                                      roi_id=roi_id)
            self.roi_list.append(roi)
            for ir in range(mask.shape[0]):
                rr = origin[0]+ir
                for ic in range(mask.shape[1]):
                    cc = origin[1]+ic
                    if mask[ir, ic]:
                        self.roi_pixels[rr, cc] = True

        return seed_list

    def run(self,
            roi_output: pathlib.Path,
            seed_output: Optional[pathlib.Path] = None,
            plot_output: Optional[pathlib.Path] = None) -> None:
        """
        Actually perform the work of detecting ROIs in the video

        Parameters
        ----------
        roi_output: pathlib.Path
            Path to the JSON file where discovered ROIs will be recorded

        seed_output: Optional[pathlib.Path]
            If not None, the path where the seeds for ROI discovery at
            each iteration will be written out in a JSONized dict.
            (default: None)

        plot_output: Optional[pathlib.Path]
            If not None, the path where a plot comparing the seed image
            with the discovered ROIs will be written.
            (default: None)

        Returns
        -------
        None

        Notes
        -----
        ROIs are discovered as follows

        1) Consider all pixels not currently assigned to ROIs.
        Using the image derived from graph_input, find all of
        the peaks that are 2 sigma brighter than the median of
        those pixels. Seed an ROI around each of these peaks (as
        peaks are selected their neighborhoods are masked out so
        that candidate peaks do not cluster)

        2) Feed each seed to the PotentialROI sub-class specified
        in init. Use the algorithm implemented by that class's
        get_mask method to grow the ROI.

        3) Collect all discovered ROI pixels into one place. As long
        as pixels are added to the set of ROI pixels, return to (1)
        and continue.
        """

        t0 = time.time()

        img_data = graph_to_img(self._graph_input,
                                attribute=self._attribute)

        logger.info(f'read in image data from {str(self._graph_input)}')

        with h5py.File(self._video_input, 'r') as in_file:
            video_data = in_file['data'][()]
        logger.info(f'read in video data from {str(self._video_input)}')

        if seed_output is not None:
            seed_record = {}

        # list of discovered ROIs
        self.roi_list = []

        # running unique identifier of ROIs
        self.roi_id = -1

        # all pixels that have been flagged as belonging
        # to an ROI
        self.roi_pixels = np.zeros(img_data.shape, dtype=bool)

        keep_going = True
        i_iteration = 0

        while keep_going:

            n_roi_0 = self.roi_pixels.sum()
            roi_seeds = self._run(img_data, video_data)
            n_roi_1 = self.roi_pixels.sum()

            duration = time.time()-t0

            msg = f'Completed iteration with {len(roi_seeds)} ROIs '
            msg += f'after {duration:.2f} seconds; '
            msg += f'{n_roi_1} total ROI pixels'
            logger.info(msg)

            if seed_output is not None:
                seed_record[i_iteration] = roi_seeds

            i_iteration += 1
            if n_roi_1 <= n_roi_0:
                keep_going = False

        logger.info('finished iterating on ROIs')

        if seed_output is not None:
            logger.info(f'writing {str(seed_output)}')
            with open(seed_output, 'w') as out_file:
                out_file.write(json.dumps(seed_record, indent=2))

        logger.info(f'writing {str(roi_output)}')
        with open(roi_output, 'w') as out_file:
            out_file.write(json.dumps(self.roi_list, indent=2))

        if plot_output is not None:
            logger.info(f'writing {str(plot_output)}')
            create_roi_plot(plot_output, img_data, self.roi_list)

        duration = time.time()-t0
        logger.info(f'Completed segmentation in {duration:.2f} seconds')
        return None
