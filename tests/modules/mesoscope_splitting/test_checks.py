import json
import pytest
import numpy as np
from pathlib import Path
from contextlib import nullcontext

import ophys_etl.modules.mesoscope_splitting.checks as checks


@pytest.fixture
def mock_tiff_list(request, tmpdir):
    mtl = []
    for i, vals in enumerate(request.param["header_values"]):
        tpath = tmpdir / f"tmp_{i}.tiff"
        mydict = {
                "RoiGroups": {
                    "imagingRoiGroup": {
                        "rois": [{"discretePlaneMode": i} for i in vals]}}}
        with open(tpath, "w") as f:
            json.dump(mydict, f)
        mtl.append(tpath)
    yield mtl


@pytest.mark.parametrize(
        "mock_tiff_list, roi_indices, context",
        [
            (
                # the expected consistent case
                {"header_values": [[0, 1, 1, 1],
                                   [1, 0, 1, 1],
                                   [1, 1, 0, 1],
                                   [1, 1, 1, 0]]},
                [[0], [1], [2], [3]],
                nullcontext()
                ),
            (
                # the first tiff index is mismatched between json and header
                {"header_values": [[1, 0, 1, 1],
                                   [1, 0, 1, 1],
                                   [1, 1, 0, 1],
                                   [1, 1, 1, 0]]},
                [[0], [1], [2], [3]],
                pytest.raises(ValueError, match=r"expected the input json*")
                ),
            (
                # the first header has multiple targeted regions (=0)
                {"header_values": [[0, 1, 0, 1],
                                   [1, 0, 1, 1],
                                   [1, 1, 0, 1],
                                   [1, 1, 1, 0]]},
                [[0], [1], [2], [3]],
                pytest.raises(ValueError, match=r"expected each tiff header*")
                ),
            (
                # the first input json entry has different target region
                # values for the 2 experiments
                {"header_values": [[0, 1, 1, 1],
                                   [1, 0, 1, 1],
                                   [1, 1, 0, 1],
                                   [1, 1, 1, 0]]},
                [[0, 1], [1], [2], [3]],
                pytest.raises(ValueError,
                              match=r"expected each tiff to target*")
                ),
            (
                # combination of problems across different tiffs
                # accumulate the errors
                {"header_values": [[0, 1, 1, 1],
                                   [1, 0, 1, 1],
                                   [1, 1, 0, 1],
                                   [1, 0, 1, 0]]},
                [[0, 1], [1], [2], [3]],
                pytest.raises(checks.MultiException)
                ),
            ], indirect=["mock_tiff_list"])
def test_splitting_consistency_check(mock_tiff_list, roi_indices,
                                     context, monkeypatch):
    def mock_tiff_header_reader(arg):
        with open(arg, "r") as f:
            j = json.load(f)
        return (None, j)
    monkeypatch.setattr(checks, "tiff_header_data", mock_tiff_header_reader)
    mylist = [checks.ConsistencyInput(tiff=tiff, roi_index=roi_index)
              for tiff, roi_index in zip(mock_tiff_list, roi_indices)]
    with context:
        checks.splitting_consistency_check(mylist)


class MockMesoscopeTiff():
    def __init__(self, path):
        self._source = path
        return

    @property
    def plane_scans(self):
        return np.array([-25, 15, 80, 90, 15, 120, 130, 180])


def test_check_for_repeated_planes(tmpdir):
    placeholder = Path(tmpdir / "tmp.tiff")
    with pytest.raises(ValueError,
                       match=f"{placeholder.name} has a repeated*"):
        checks.check_for_repeated_planes(MockMesoscopeTiff(placeholder))
