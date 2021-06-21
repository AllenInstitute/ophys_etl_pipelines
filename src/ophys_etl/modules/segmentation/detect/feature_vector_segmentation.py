from typing import Optional, List, Tuple
import numpy as np
import multiprocessing
import multiprocessing.managers
import h5py
import pathlib
import time
import json
from matplotlib.figure import Figure

from ophys_etl.types import ExtractROI
from ophys_etl.modules.segmentation.detect.feature_vector_rois import (
    PearsonFeatureROI)
from ophys_etl.modules.segmentation.qc_utils.roi_utils import create_roi_plot
from ophys_etl.modules.segmentation.graph_utils.conversion import graph_to_img
from ophys_etl.modules.segmentation.seed.seeder import \
        BatchImageMetricSeeder
from ophys_etl.modules.segmentation.qc.seed import add_seeds_to_axes

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

    seeder_args: dict
        passed to BatchImageMetricSeeder.__init__()

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
                 seeder_args: dict,
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

        self.seeder = BatchImageMetricSeeder(**seeder_args)
        self.seeder.select_seeds(self._graph_img, sigma=None)

        with h5py.File(self._video_input, 'r') as in_file:
            movie_shape = in_file["data"].shape
            if movie_shape[1:] != self._graph_img.shape:
                msg = f'movie shape: {movie_shape}\n'
                msg += f'img shape: {self._graph_img.shape}'
                raise RuntimeError(msg)
        self.movie_shape = movie_shape

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

        # NOTE: we should rewrite run() and _run() so that they can
        # use the parallel seed iterator like
        # ```
        # for seed_list in self.seeder:
        #     <farm out processes>
        # ```
        # for now:
        try:
            seed_list = next(self.seeder)
        except StopIteration:
            seed_list = []

        slop = 20

        logger.info(f'got {len(seed_list)} seeds')

        # ROIs can be grown independently of each other;
        # farm each seed out to an independent process
        p_list = []
        mgr = multiprocessing.Manager()
        mgr_dict = mgr.dict()
        for i_seed, seed in enumerate(seed_list):
            self.roi_id += 1
            r0 = int(max(0, seed[0] - slop))
            r1 = int(min(self.movie_shape[0], seed[0] + slop))
            c0 = int(max(0, seed[1] - slop))
            c1 = int(min(self.movie_shape[1], seed[1] + slop))

            mask = self.roi_pixels[r0:r1, c0:c1]

            video_data_subset = video_data[:, r0:r1, c0:c1]

            # NOTE: eventually, get rid of ROISeed
            # rationale: seeding produces seeds (coordinates), this object
            # specifies a growth region, which should be a "segment",
            # i.e. "detect" role.
            this_seed = ROISeed(center=seed,
                                rows=[r0, r1],
                                cols=[c0, c1])

            p = multiprocessing.Process(target=_get_roi,
                                        args=(this_seed,
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
                        # make sure the seeder does not
                        # supply new seeds that are in these ROIs
                        self.seeder.exclude_pixels({(rr, cc)})

        return seed_list

    def run(self,
            roi_output: pathlib.Path,
            seed_output: Optional[pathlib.Path] = None,
            plot_output: Optional[pathlib.Path] = None,
            seed_plot_output: Optional[pathlib.Path] = None,
            ) -> None:
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

            roi_seeds = self._run(img_data, video_data)

            # NOTE: this change lets the seeder/iterator control
            # the stopping condition of segmentation. I.e. when seeds
            # are exhausted. run() and _run() should be rewritten to use
            # the iterator in a more canonical way.
            if len(roi_seeds) == 0:
                break

            duration = time.time()-t0

            msg = f'Completed iteration with {len(roi_seeds)} ROIs '
            msg += f'after {duration:.2f} seconds; '
            msg += f'{self.roi_pixels.sum()} total ROI pixels'
            logger.info(msg)

            if seed_output is not None:
                seed_record[i_iteration] = roi_seeds

            i_iteration += 1

        logger.info('finished iterating on ROIs')

        if seed_output is not None:
            with h5py.File(seed_output, "w") as f:
                self.seeder.log_to_h5_group(f)
            logger.info(f'wrote {str(seed_output)}')

            if seed_plot_output is not None:
                f = Figure(figsize=(8, 8))
                axes = f.add_subplot(111)
                add_seeds_to_axes(f, axes, seed_h5_path=seed_output)
                f.savefig(seed_plot_output)
                logger.info(f'wrote {seed_plot_output}')

        logger.info(f'writing {str(roi_output)}')
        with open(roi_output, 'w') as out_file:
            out_file.write(json.dumps(self.roi_list, indent=2))

        if plot_output is not None:
            logger.info(f'writing {str(plot_output)}')
            create_roi_plot(plot_output, img_data, self.roi_list)

        duration = time.time()-t0
        logger.info(f'Completed segmentation in {duration:.2f} seconds')
        return None
