from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import argschema
import json
import h5py
import pathlib
import numpy as np

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.merge.roi_utils import (
    ophys_roi_to_extract_roi)

from ophys_etl.modules.segmentation.qc_utils.roi_comparison_utils import (
    roi_list_from_file)


class ROIBaseFilter(ABC):

    _reason = None  # reason ROI is marked as invalid

    @abstractmethod
    def is_roi_valid(self, roi: OphysROI) -> bool:
        raise NotImplementedError()

    def do_filtering(
            self,
            roi_list: List[OphysROI]) -> Dict[str, List[OphysROI]]:

        valid_roi = []
        invalid_roi = []
        for roi in roi_list:
            if self.is_roi_valid(roi):
                valid_roi.append(roi)
            else:
                new_roi = OphysROI(
                              x0=roi.x0,
                              y0=roi.y0,
                              height=roi.height,
                              width=roi.width,
                              mask_matrix=roi.mask_matrix,
                              roi_id=roi.roi_id,
                              valid_roi=False)
                invalid_roi.append(new_roi)

        return {'valid_roi': valid_roi,
                'invalid_roi': invalid_roi}

    @property
    def reason(self):
        if self._reason is None:
            msg = "self._reason not defined for class "
            msg += f"{type(self)}"
            raise NotImplementedError(msg)
        return self._reason


class ROIAreaFilter(ROIBaseFilter):

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
    def max_area(self):
        return self._max_area

    @property
    def min_area(self):
        return self._min_area

    def is_roi_valid(self, roi: OphysROI) -> bool:
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
        roi_list = roi_list_from_file(pathlib.Path(self.args['roi_input']))

        this_filter = self.get_filter()

        results = this_filter.do_filtering(roi_list)

        reason = ' -- '.join((this_filter.reason,
                              self.args['pipeline_stage']))
        log_invalid_rois(results['invalid_roi'],
                         reason,
                         pathlib.Path(self.args['roi_log_path']))

        new_roi_list = [ophys_roi_to_extract_roi(roi)
                        for roi in results['valid_roi']]
        new_roi_list += [ophys_roi_to_extract_roi(roi)
                         for roi in results['invalid_roi']]

        with open(self.args['roi_output'], 'w') as out_file:
            json.dump(new_roi_list, out_file, indent=2)
