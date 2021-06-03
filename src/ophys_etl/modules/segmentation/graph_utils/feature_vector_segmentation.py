from typing import Optional, List, Tuple
import numpy as np
import multiprocessing
import multiprocessing.managers
import h5py
import pathlib
import time
import json

from ophys_etl.types import ExtractROI

from ophys_etl.modules.segmentation.graph_utils.feature_vector_rois import (
    PearsonFeatureROI)

from ophys_etl.modules.segmentation.graph_utils.plotting import (
    create_roi_plot,
    graph_to_img)

import logging

import sys
if sys.version_info >= (3, 8):
    from typing import TypedDict
else:
    from typing_extensions import TypedDict


logger = logging.getLogger(__name__)
logging.captureWarnings(True)
logging.basicConfig(level=logging.INFO)


class ROISeed(TypedDict):
    center: Tuple[int, int]
    rows: Tuple[int, int]
    cols: Tuple[int, int]


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
               n_sigma: int = 2) -> List[ROISeed]:
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
    seeds: List[ROISeed]
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

            obj = ROISeed(center=(int(pixel[0]), int(pixel[1])),
                          rows=(int(rowmin), int(rowmax)),
                          cols=(int(colmin), int(colmax)))

            output.append(obj)
            for irow in range(rowmin, rowmax):
                for icol in range(colmin, colmax):
                    ii = np.ravel_multi_index((irow, icol), shape)
                    img_masked.mask[ii] = True

    return output


def _get_roi(seed_obj: ROISeed,
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
    seed_obj: ROISeed
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
    valid = np.argwhere(mask)
    row0 = valid[:, 0].min()
    row1 = valid[:, 0].max() + 1
    col0 = valid[:, 1].min()
    col1 = valid[:, 1].max() + 1

    new_mask = mask[row0:row1, col0:col1]
    roi = ExtractROI(id=roi_id,
                     x=int(origin[1]+col0),
                     y=int(origin[0]+row0),
                     width=int(col1-col0),
                     height=int(row1-row0),
                     valid=False,
                     mask=[i.tolist() for i in new_mask])
    return roi


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
                                       attribute_name=attribute)

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

    def _load_video(self, video_path: pathlib.Path):
        with h5py.File(video_path, 'r') as in_file:
            video_data = in_file['data'][()]
        return video_data

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
                                attribute_name=self._attribute)

        logger.info(f'read in image data from {str(self._graph_input)}')

        video_data = self._load_video(self._video_input)
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
