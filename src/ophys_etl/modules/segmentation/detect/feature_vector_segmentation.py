from typing import Optional, List, Tuple
import numpy as np
import multiprocessing
import multiprocessing.managers
import h5py
import pathlib
import time
import json
import matplotlib

from ophys_etl.types import ExtractROI
from ophys_etl.modules.segmentation.detect.feature_vector_rois import (
    PearsonFeatureROI)
from ophys_etl.modules.segmentation.graph_utils.conversion import graph_to_img
from ophys_etl.modules.segmentation.seed.seeder import \
        BatchImageMetricSeeder
from ophys_etl.modules.segmentation.qc.seed import add_seeds_to_axes
from ophys_etl.modules.segmentation.qc.detect import roi_metric_qc_plot

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


def _is_roi_at_edge(origin: Tuple[int, int],
                    fov_shape: Tuple[int, int],
                    mask: np.ndarray) -> bool:
    """
    Check if an ROI grew to the edge of its thumbnail (in which
    case the ROI should be attempted again with a larger thumbnail)

    Parameters
    ----------
    origin: Tuple[int, int]
        (row, col) coordinates of the ROI thumbnail's origin

    fov_shape: Tuple[int, int]
        (nrows, ncols) of the full video field of view

    mask: np.ndarray
        2D array of booleans encoding the ROI's mask

    Returns
    -------
    at_edge: boolean
       True if there are any pixels marked 'True' in the extremal
       edge of the thumbnail *and* there is room for the thumbnail
       to grow beyond that extremal edge (i.e. if the origin is at
       (0, 0) and there are True pixels in the first column, there
       are no columns available to the left of the origin, so
       return False). Return False otherwise.
    """

    shape = mask.shape

    first_row = (origin[0] > 0) & mask[0, :].any()
    last_row = (origin[0]+shape[0] < fov_shape[0]) & mask[-1, :].any()
    first_col = (origin[1] > 0) & mask[:, 0].any()
    last_col = (origin[1]+shape[1] < fov_shape[1]) & mask[:, -1].any()

    return first_row | last_row | first_col | last_col


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
                     valid=True,
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
             video_data: np.ndarray) -> List[dict]:
        """
        Run one iteration of ROI detection

        Parameters
        ----------
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

        # in case we end up retrying ROIs, make sure we can
        # grow their available thumbnails
        seed_to_slop = dict()

        # NOTE: we should rewrite run() and _run() so that they can
        # use the parallel seed iterator like
        # ```
        # for seed_list in self.seeder:
        #     <farm out processes>
        # ```
        # for now:

        if len(self.roi_to_retry) > 0:
            # if there are ROIs from a previous iteration that filled
            # their thumbnail, try those again first with a larger
            # thumbnail
            seed_list = []
            for roi in self.roi_to_retry:
                seed_list.append(roi['seed'])
                seed_to_slop[roi['seed']] = 3*roi['slop']//2

            self.roi_to_retry = []
        else:
            # run with new seeds from the seeder
            try:
                seed_list = next(self.seeder)
            except StopIteration:
                seed_list = []

        default_slop = 20
        max_slop = 40

        # lookup from ROI ID to seed and size of ROI
        # thumbnail
        roi_inputs = dict()

        logger.info(f'got {len(seed_list)} seeds')

        # ROIs can be grown independently of each other;
        # farm each seed out to an independent process
        p_list = []
        mgr = multiprocessing.Manager()
        mgr_dict = mgr.dict()
        for i_seed, seed in enumerate(seed_list):
            if self.roi_pixels[seed[0], seed[1]]:
                continue

            self.roi_id += 1
            slop = seed_to_slop.get(seed, default_slop)

            roi_inputs[self.roi_id] = {'seed': seed,
                                       'slop': slop}

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
            at_edge = _is_roi_at_edge(origin,
                                      video_data.shape[1:],
                                      mask)
            if at_edge and roi_inputs[roi_id]['slop'] < max_slop:
                self.roi_to_retry.append(roi_inputs[roi_id])
                continue

            roi = convert_to_lims_roi(origin,
                                      mask,
                                      roi_id=roi_id)
            if mask.sum() > 1:
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
            qc_output: pathlib.Path,
            plot_output: Optional[pathlib.Path] = None,
            seed_plot_output: Optional[pathlib.Path] = None,
            ) -> None:
        """
        Actually perform the work of detecting ROIs in the video

        Parameters
        ----------
        roi_output: pathlib.Path
            Path to the JSON file where discovered ROIs will be recorded

        qc_output: pathlib.Path
            the path where the qc results will be written

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

        # list to keep track of ROIs that fill their thumbnail
        self.roi_to_retry = []

        t0 = time.time()

        logger.info(f'read in image data from {str(self._graph_input)}')

        with h5py.File(self._video_input, 'r') as in_file:
            video_data = in_file['data'][()]
        logger.info(f'read in video data from {str(self._video_input)}')

        seed_record = {}

        # list of discovered ROIs
        self.roi_list = []

        # running unique identifier of ROIs
        self.roi_id = -1

        # all pixels that have been flagged as belonging
        # to an ROI
        self.roi_pixels = np.zeros(self._graph_img.shape, dtype=bool)

        keep_going = True
        i_iteration = 0

        while keep_going:

            roi_seeds = self._run(video_data)

            # NOTE: this change lets the seeder/iterator control
            # the stopping condition of segmentation. I.e. when seeds
            # are exhausted. run() and _run() should be rewritten to use
            # the iterator in a more canonical way.
            if len(roi_seeds) == 0:
                break

            duration = time.time()-t0
            n_valid_pix = 0
            for roi in self.roi_list:
                m = np.array(roi['mask'])
                n_valid_pix += m.sum()

            msg = f'Completed iteration with {len(roi_seeds)} ROIs '
            msg += f'after {duration:.2f} seconds; '
            msg += f'{self.roi_pixels.sum()} total ROI pixels; '
            msg += f'{len(self.roi_list)} valid ROIs ({n_valid_pix}) pixels'
            logger.info(msg)

            seed_record[i_iteration] = roi_seeds

            i_iteration += 1

        logger.info('finished iterating on ROIs')

        # log seeder to hdf5 QC output
        with h5py.File(qc_output, "a") as f:
            self.seeder.log_to_h5_group(f)
        logger.info(f'wrote seeding QC to {str(qc_output)}')

        # log detection to hdf5 QC ouput
        with h5py.File(qc_output, "a") as h5file:
            group = h5file.create_group("detect")
            group.create_dataset("metric_image", data=self._graph_img)
            group.create_dataset("attribute",
                                 data=self._attribute.encode("utf-8"))

        # create a plot from the seeder QC output
        if seed_plot_output is not None:
            fig = matplotlib.figure.Figure(figsize=(8, 8))
            axes = fig.add_subplot(111)
            with h5py.File(qc_output, "r") as f:
                add_seeds_to_axes(fig, axes, seed_h5_group=f["seed"])
            fig.savefig(seed_plot_output)
            logger.info(f'wrote {seed_plot_output}')

        logger.info(f'writing {str(roi_output)}')
        with open(roi_output, 'w') as out_file:
            out_file.write(json.dumps(self.roi_list, indent=2))

        if plot_output is not None:
            # TODO: store ROIs in QC output hdf5
            figure = matplotlib.figure.Figure(figsize=(10, 10))
            with h5py.File(qc_output, "r") as f:
                group = f["detect"]
                roi_metric_qc_plot(
                        figure=figure,
                        metric_image=group["metric_image"][()],
                        attribute=group["attribute"][()].decode("utf-8"),
                        roi_list=self.roi_list)
            figure.tight_layout()
            figure.savefig(plot_output)
            logger.info(f'wrote {plot_output}')

        duration = time.time()-t0
        logger.info(f'Completed segmentation in {duration:.2f} seconds')
        return None
