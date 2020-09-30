import pytest
import numpy as np
from scipy.sparse import coo_matrix
from marshmallow import ValidationError

from ophys_etl.schemas import SparseAndDenseROISchema


class TestSparseAndDenseROISchema:
    def test_schema_makes_coos(self, rois_fixture):
        expected = [coo_matrix(np.array(r["mask_matrix"]))
                    for r in rois_fixture.rois]
        actual = SparseAndDenseROISchema(many=True).load(rois_fixture.rois)
        for ix, e_roi in enumerate(expected):
            np.testing.assert_array_equal(
                e_roi.toarray(), actual[ix]["coo_roi"].toarray())

    @pytest.mark.parametrize(
        "data, expected_coo",
        [
            (
                {"x": 1, "y": 1, "height": 3, "width": 1,
                 "mask_matrix": [[True], [False], [True]]},
                coo_matrix(np.array([[1], [0], [1]]))
            ),
            (   # Empty
                {"x": 100, "y": 34, "height": 0, "width": 0,
                 "mask_matrix": []},
                coo_matrix(np.array([]))
            ),
        ]
    )
    def test_coo_roi_dump_single(self, data, expected_coo, rois_fixture):
        """Cover the empty case, another unit test"""
        data.update(rois_fixture.additional_data)
        data.update({"exclusion_labels": [], "id": 42})
        rois = SparseAndDenseROISchema().load(data)
        expected_data = data.copy()
        expected_data.update({"coo_roi": expected_coo})
        np.testing.assert_array_equal(
            expected_data["coo_roi"].toarray(), rois["coo_roi"].toarray())

    @pytest.mark.parametrize(
        "data",
        [
            {"x": 1, "y": 1, "height": 5, "width": 1,    # height
             "mask_matrix": [[True], [False], [True]]},
            {"x": 1, "y": 1, "height": 3, "width": 2,    # width
             "mask_matrix": [[True], [False], [True]]},
        ]
    )
    def test_coo_roi_dump_raise_error_mismatch_dimensions(self, data,
                                                          rois_fixture):
        with pytest.raises(ValidationError) as e:
            data.update(rois_fixture.additional_data)
            data.update({"exclusion_labels": [], "id": 42})
            SparseAndDenseROISchema().load(data)
        assert "Data in mask matrix did not correspond" in str(e.value)

    def test_schema_warns_empty(self, rois_fixture):
        rois_fixture.rois[0]["height"] = 0
        rois_fixture.rois[0]["width"] = 0
        with pytest.warns(UserWarning) as record:
            SparseAndDenseROISchema(many=True).load(rois_fixture.rois)
        assert len(record) == 1
        assert "Input data contains empty ROI" in str(record[0].message)

    def test_schema_errors_shape_mismatch(self, rois_fixture):
        rois_fixture.rois[0]["height"] = 100
        rois_fixture.rois[0]["width"] = 29
        with pytest.raises(ValidationError) as e:
            SparseAndDenseROISchema(many=True).load(rois_fixture.rois)
        assert "Data in mask matrix did not correspond" in str(e.value)
