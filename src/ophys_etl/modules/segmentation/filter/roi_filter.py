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
    ophys_roi_list_from_deserialized)
from ophys_etl.modules.segmentation.processing_log import \
    SegmentationProcessingLog


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
        processing_log = SegmentationProcessingLog(self.args["log_path"])
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
        processing_log.log_filter(rois=rois,
                                  filter_ids=invalid_ids,
                                  filter_reason=reason,
                                  group_name="filter")
        self.logger.info(f'added group {processing_log.get_last_group()} '
                         f'to {processing_log.path}')
