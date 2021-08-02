from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
import argschema
import h5py
import pathlib
import copy
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.segmentation.utils.roi_utils import (
    ophys_roi_to_extract_roi,
    ophys_roi_list_from_deserialized,
    background_mask_from_roi_list)
from ophys_etl.modules.segmentation.processing_log import \
    SegmentationProcessingLog

from ophys_etl.modules.segmentation.filter.filter_utils import (
    mean_metric_from_roi,
    median_metric_from_roi,
    z_vs_background_from_roi)


class ROIBaseFilter(ABC):
    """
    Abstract base class for ROI filtering mechanisms.

    Sub-classes of this calss need to implement
    self.is_roi_valid which accepts an OphysROI and returns
    a boolean indicating whether or not that ROI is accepted
    as valid according to the filter's criterion.

    The __init__ for the sub-class also needs to define
    self._reason, a string summarizing what the filter does
    so that users can know why their ROIs were flagged as
    invalid.
    """

    @abstractmethod
    def is_roi_valid(self, roi: OphysROI) -> bool:
        """
        Determines if an ROI passes the filter

        Parameters
        ----------
        roi: OphysROI

        Returns
        -------
        bool
            True if the ROI is determined as valid;
            False otherwise
        """
        raise NotImplementedError()

    def _pre_processing(self,
                        roi_list: List[OphysROI]) -> List[OphysROI]:
        """
        Method that can be overloaded to do any pre-processing
        of roi_list as a whole before each ROI is passed to
        self.is_roi_valid

        Parameters
        ----------
        roi_list: List[OphysROI]

        Returns
        -------
        roi_list: List[OphysROI]
            The list of ROIs after preprocessing has been
            applied (default for base class performs no
            actual operation)
        """
        return roi_list

    def do_filtering(
            self,
            roi_list: List[OphysROI]) -> Dict[str, List[OphysROI]]:
        """
        Loop over a list of ROIs, testing each against the filter's
        criterion. Returns a dict in which 'valid_roi' points to
        a list of ROIs that passed the cut and 'invalid_roi' points
        to a list of ROIs that did not. Invalid ROIs will be
        re-instantiated with valid_roi=False

        Parameters
        ----------
        roi_list: List[OphysROI]

        Returns
        -------
        Dict[str, List[OphysROI]]
        """
        roi_list = self._pre_processing(roi_list)

        valid_roi = []
        invalid_roi = []
        for roi in roi_list:
            new_roi = copy.deepcopy(roi)
            if self.is_roi_valid(roi):
                valid_roi.append(new_roi)
            else:
                new_roi.valid_roi = False
                invalid_roi.append(new_roi)

        return {'valid_roi': valid_roi,
                'invalid_roi': invalid_roi}

    @property
    def reason(self) -> str:
        """
        Return a string encapsulating the reason an ROI
        would be flagged invalid by this filter.
        """
        if not hasattr(self, '_reason'):
            msg = "self._reason not defined for class "
            msg += f"{type(self)}\n"
            msg += "must set it in __init__"
            raise NotImplementedError(msg)
        return self._reason


class ROIAreaFilter(ROIBaseFilter):
    """
    A sub-class of ROIBaseFilter that accepts ROIs if their
    areas are min_area <= area <= max_area

    Parameters
    ----------
    min_area: Optional[int]
        Default None

    max_area: Optional[int]
         Default None

    Note
    ----
    Any limit that is None is ignored. If both limits are
    None, an error is raised.
    """

    def __init__(self,
                 min_area: Optional[int] = None,
                 max_area: Optional[int] = None):

        self._reason = 'area'

        if max_area is None and min_area is None:
            msg = "Both max_area and min_area are None; "
            msg += "you must specify at least one"
            raise RuntimeError(msg)

        self._max_area = max_area
        self._min_area = min_area

    @property
    def max_area(self) -> Union[int, None]:
        """
        Maximum allowable area for a valid ROI
        """
        return self._max_area

    @property
    def min_area(self) -> Union[int, None]:
        """
        Minimum allowable area for a valid ROI
        """
        return self._min_area

    def is_roi_valid(self, roi: OphysROI) -> bool:
        """
        Determines if an ROI satisifies
        min_area <= roi.area <= max_area

        Parameters
        ----------
        roi: OphysROI

        Returns
        -------
        bool
            True if the ROI is determined as valid;
            False otherwise
        """

        if self.max_area is not None:
            if roi.area > self.max_area:
                return False
        if self.min_area is not None:
            if roi.area < self.min_area:
                return False
        return True


class ROIMetricStatFilter(ROIBaseFilter):
    """
    A sub-class of ROIBaseFilter that filters ROIs on some statistic
    computed from a metric image.

    Parameters
    ----------
    metric_img: np.ndarray
        The metric image from which to compute the summary statistic

    metric_data: str
        The name of the metric stat to be used.
        Currently accepted values are 'mean' and 'median'.

    min_metric: Optional[float]
        The minimum allowed value of the summary statistic in
        a valid ROI (default: None)

    max_metric: Optional[float]
        The maximum allowed value of the summary statistic in
        a valid ROI (default: None)

    Note
    ----
    Any limit that is None is ignored. If both limits are
    None, an error is raised.
    """

    def __init__(self,
                 metric_img: np.ndarray,
                 metric_stat: str,
                 min_metric: Optional[float] = None,
                 max_metric: Optional[float] = None):

        if metric_stat == 'mean':
            self._stat_method = mean_metric_from_roi
        elif metric_stat == 'median':
            self._stat_method = median_metric_from_roi
        else:
            msg = "ROIMetricStatFilter only knows how to compute "
            msg += "'mean' and 'median'; you gave "
            msg += f"metric_stat='f{metric_stat}'"
            raise ValueError(msg)

        if max_metric is None and min_metric is None:
            msg = "Both max_metric and min_metric are None; "
            msg += "you must specify at least one"
            raise RuntimeError(msg)

        self._max_metric = max_metric
        self._min_metric = min_metric

        self.img = np.copy(metric_img)
        self._reason = f'{metric_stat} ('
        if min_metric is not None:
            self._reason += f' min: {min_metric: .2e};'
        if max_metric is not None:
            self._reason += f' max: {max_metric: .2e}'
        self._reason += ' )'

    @property
    def max_metric(self) -> Union[float, None]:
        return self._max_metric

    @property
    def min_metric(self) -> Union[float, None]:
        return self._min_metric

    def is_roi_valid(self, roi: OphysROI) -> bool:
        metric_value = self._stat_method(roi, self.img)
        if self.max_metric is not None:
            if metric_value > self.max_metric:
                return False
        if self.min_metric is not None:
            if metric_value < self.min_metric:
                return False
        return True


class ZvsBackgroundFilter(ROIBaseFilter):
    """
    A sub-class of ROIBaseFilter that filters ROIs based on the
    z-score of their mean pixel value relative to the local
    background

    Parameters
    ----------
    metric_img: np.ndarray
        The metric image from which to compute the summary statistic

    min_z: float
        The minimum z-score allowed for a valid ROI

    n_background_factor: int
        The the factor to multiply roi.area by when requesting
        background pixels

    n_background_min: int
        The minimum number of background pixels to use
        (in case roi.area is very small)
    """

    def __init__(self,
                 metric_img: np.ndarray,
                 min_z: float,
                 n_background_factor: int,
                 n_background_min: int):
        self._img = np.copy(metric_img)
        self._min_z = min_z
        self._n_background_factor = n_background_factor
        self._n_background_min = n_background_min
        self._background_mask = None
        self._reason = "z-score vs background pixels"

    @property
    def img(self) -> np.ndarray:
        return self._img

    @property
    def min_z(self) -> float:
        return self._min_z

    @property
    def n_background_factor(self) -> int:
        return self._n_background_factor

    @property
    def n_background_min(self) -> int:
        return self._n_background_min

    @property
    def background_mask(self) -> np.ndarray:
        if self._background_mask is None:
            msg = "self._background_mask is None; "
            msg += "cannot filter ROIs based on z-score "
            msg += "versus background"
            raise RuntimeError(msg)
        return self._background_mask

    def _pre_processing(self, roi_list: List[OphysROI]) -> List[OphysROI]:
        """
        Use the full list of ROI to construct a mask indicating which
        pixels are not a part of any ROIs

        Parameters
        ----------
        roi_list: List[OphysROI]

        Returns
        -------
        roi_list: List[OphysROI]

        Notes
        -----
        This method does not alter roi_list. It does create
        self._background_mask, an np.ndarray marked as True
        for all background pixels
        """
        self._background_mask = background_mask_from_roi_list(
                                    roi_list,
                                    self.img.shape)
        return roi_list

    def is_roi_valid(self, roi: OphysROI) -> bool:
        n_bckgd = max(self.n_background_min,
                      self.n_background_factor*roi.area)
        z_score = z_vs_background_from_roi(
                        roi,
                        self.img,
                        self.background_mask,
                        n_desired_background=n_bckgd)
        if z_score < self.min_z:
            return False
        return True


def log_invalid_rois(roi_list: List[OphysROI],
                     reason: str,
                     log_path: pathlib.Path) -> None:
    """
    Write the ROI IDs of invalid ROIs to the HDF5 log,
    along with a brief summary of why they are invalid.

    This method will record the ROI IDs in
    filter_log/invalid_roi_id
    and the reasons in
    filter_log/reason

    If there is already a log of invalid ROIs in the
    HDF5 file, these ROIs will be appended to it.

    Parameters
    ----------
    roi_list: List[OphysROI]
        List of the invalid ROIs

    reason: str
        Brief summary of why these ROIs are invalid

    log_path: pathlib.Path
        Path to the HDF5 file being logged
    """

    group_name = 'filter_log'
    old_roi_id = []
    old_reasons = []
    with h5py.File(log_path, 'a') as log_file:
        if f'{group_name}/reason' in log_file.keys():
            old_roi_id = list(log_file[f'{group_name}/invalid_roi_id'][()])
            old_reasons = list(log_file[f'{group_name}/reason'][()])
            del log_file[f'{group_name}/invalid_roi_id']
            del log_file[f'{group_name}/reason']
        roi_id = old_roi_id + [roi.roi_id for roi in roi_list]
        reasons = old_reasons + [reason.encode('utf-8')]*len(roi_list)
        log_file.create_dataset(f'{group_name}/invalid_roi_id',
                                data=np.array(roi_id))
        log_file.create_dataset(f'{group_name}/reason', data=np.array(reasons))
    return None


class FilterRunnerBase(argschema.ArgSchemaParser):
    """
    Arbitrary ROI filter runners are implemented by sub-classing
    this class and

    1) defining the default_schema
    2) implementing self.get_filter() which returns the appropriately
    instantiated sub-class of ROIBaseFilter
    """
    default_schema = None

    def get_filter(self):
        msg = "Need to implement get_filter() for "
        msg += f"{type(self)}"
        raise NotImplementedError(msg)

    def run(self):
        # get the ROIs to filter
        processing_log = SegmentationProcessingLog(self.args["log_path"],
                                                   read_only=True)
        original_roi_list = processing_log.get_rois_from_group(
                group_name=self.args["rois_group"])
        ophys_roi_list = ophys_roi_list_from_deserialized(original_roi_list)

        # perform filter
        this_filter = self.get_filter()
        results = this_filter.do_filtering(ophys_roi_list)

        # munge
        invalid_rois = [ophys_roi_to_extract_roi(i)
                        for i in results["invalid_roi"]]
        invalid_ids = [i["id"] for i in invalid_rois]
        rois = [ophys_roi_to_extract_roi(i)
                for i in results["valid_roi"]]
        rois.extend(invalid_rois)

        reason = ' -- '.join((this_filter.reason,
                              self.args['pipeline_stage']))

        # log
        processing_log = SegmentationProcessingLog(self.args["log_path"],
                                                   read_only=False)
        processing_log.log_filter(rois=rois,
                                  roi_source_group=self.args["rois_group"],
                                  filter_ids=invalid_ids,
                                  filter_reason=reason,
                                  group_name="filter")
        self.logger.info(f'added group {processing_log.get_last_group()} '
                         f'to {processing_log.path}')
