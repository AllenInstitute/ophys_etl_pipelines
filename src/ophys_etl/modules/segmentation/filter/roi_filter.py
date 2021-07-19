from typing import List, Dict, Optional
from abc import ABC, abstractmethod

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI


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
                              mask=roi.mask_matrix,
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
                 max_area: Optional[int] = None,
                 min_area: Optional[int] = None):

        self._reason = 'area'.encode('utf-8')

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
