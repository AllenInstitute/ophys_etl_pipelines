import pytest

from ophys_etl.modules.segmentation.filter.roi_filter import (
    ROIBaseFilter)


def test_base_roi_filter():

    class DummyFilter(ROIBaseFilter):

        def is_roi_valid(self, roi):
            return {'valid_roi': [],
                    'invalid_roi': []}

    roi_filter = DummyFilter()

    with pytest.raises(NotImplementedError,
                       match="self._reason not defined"):
        roi_filter.reason
