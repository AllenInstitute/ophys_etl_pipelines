import json
import math
import os.path

import boto3
import joblib
import numpy as np
import pytest
from marshmallow import ValidationError
from moto import mock_s3
from scipy.sparse import coo_matrix
from ophys_etl.modules.classifier_inference.__main__ import (
        InferenceParser, main, _munge_traces, _munge_data,
        downsample, load_model, filter_excluded_rois)
from ophys_etl.modules.classifier_inference.schemas import (
        SparseAndDenseROISchema)


@pytest.mark.parametrize(
    "data,input_fps,output_fps,expected",
    [
        (np.ones(10,), 10, 5, np.ones(5,)),
        (np.ones(10,), 10, 3, np.ones(3,))
    ]
)
def test_downsample(data, input_fps, output_fps, expected):
    actual = downsample(data, input_fps, output_fps)
    np.testing.assert_array_equal(expected, actual)


def test_downsample_raises_error_greater_output_fps():
    """Output FPS can't be greater than input FPS"""
    with pytest.raises(
            ValueError,
            match=r'Output FPS can\'t be greater than input FPS'):
        downsample(np.arange(10), 1, 5)


def test_filter_excluded_rois(rois):
    included, excluded = filter_excluded_rois(rois)
    assert included == rois[:-1]
    assert excluded == [rois[-1]]


@pytest.mark.parametrize("roi_data, trace_file_fixture, expected", [
    # Case: ROI id order in segmentation output matches trace "roi_names" order
    ([{"id": 10}, {"id": 100}, {"id": 20}, {"id": 200}, {"id": 3}],  # roi_data
     {"trace_data": np.arange(100).reshape((5, 20)),  # trace_file_fixture
      "trace_names": ['10', '100', '20', '200', '3']},
     np.arange(100).reshape((5, 20))),  # expected

    # Case: ROI id order does not match order of trace "roi_names" (variant 1)
    ([{"id": 10}, {"id": 100}, {"id": 20}, {"id": 200}, {"id": 3}],
     {"trace_data": np.arange(100).reshape((5, 20)),
      "trace_names": ['100', '20', '10', '200', '3']},
     np.arange(100).reshape((5, 20))[[2, 0, 1, 3, 4]]),

    # Case: ROI id order does not match order of trace "roi_names" (variant 2)
    ([{"id": 3}, {"id": 20}, {"id": 10}, {"id": 200}, {"id": 100}],
     {"trace_data": np.arange(100).reshape((5, 20)),
      "trace_names": ['10', '100', '20', '200', '3']},
     np.arange(100).reshape((5, 20))[[4, 2, 0, 3, 1]]),

    # Case: ROI id order does not match order of trace "roi_names" (variant 3)
    ([{"id": 3}, {"id": 20}, {"id": 10}, {"id": 200}, {"id": 100}],
     {"trace_data": np.arange(100).reshape((5, 20)),
      "trace_names": ['100', '20', '10', '200', '3']},
     np.arange(100).reshape((5, 20))[[4, 1, 2, 3, 0]]),

], indirect=["trace_file_fixture"])
def test_munge_traces(roi_data, trace_file_fixture, expected):
    trace_file, fixture_params = trace_file_fixture
    obt = _munge_traces(roi_data, trace_file,
                        fixture_params['trace_data_key'],
                        fixture_params['trace_names_key'],
                        trace_sampling_rate=30,
                        desired_trace_sampling_rate=30)
    assert np.allclose(obt, expected)


@pytest.fixture(scope="function")
def joblib_model_fixture(tmp_path, request):
    model_path = str(tmp_path / "my_model.joblib")
    model_data = request.param.get("data", [1, 2, 3, 4])
    joblib.dump(model_data, model_path)
    return model_path, model_data


@pytest.mark.parametrize("joblib_model_fixture, test_s3_uri", [
    ({}, True),
    ({"data": "42"}, True),

    ({}, False),
    ({"data": "42"}, False)
], indirect=["joblib_model_fixture"])
def test_load_model_with_s3_uri(joblib_model_fixture, test_s3_uri):
    model_path, model_data = joblib_model_fixture

    if test_s3_uri:
        with mock_s3():
            client = boto3.client("s3")
            client.create_bucket(Bucket="mybucket",
                                 CreateBucketConfiguration={
                                     'LocationConstraint': 'us-west-2'})
            client.upload_file(model_path, "mybucket", "my_model.joblib")
            uri = "s3://mybucket/my_model.joblib"
            obt = load_model(uri)
    else:
        uri = model_path
        obt = load_model(uri)
    assert obt == model_data


class TestFeatureExtractorModule:
    def test_main_integration(self, monkeypatch, mock_model, input_data):
        """Test main module runner including smoke test for FeatureExtractor
        integration."""
        # mock out the classifier loading and predictions (in a valid way)
        def predict_fn(x):
            predictions = np.tile([True, None, False], math.ceil(len(x)/3))
            return predictions[:len(x)]

        monkeypatch.setattr("joblib.load", lambda x: mock_model(predict_fn))
        # Note for future devs -- `args=[]` is VERY important to pass when
        # testing argschema modules using VSCode and probably other IDEs.
        # If not, will exit during parsing with no good error
        # (it's ingesting the command line args passed to the test runner)
        parser = InferenceParser(input_data=input_data, args=[])
        main(parser)
        # Check outputs exist (contents are mock result)
        assert os.path.exists(parser.args["output_json"])

    def test_main_raises_value_error_unequal_predictions(
            self, monkeypatch, mock_model, input_data):
        monkeypatch.setattr(
             "joblib.load", lambda x: mock_model(lambda y: [1]))
        with pytest.raises(ValueError) as e:
            main(InferenceParser(input_data=input_data, args=[]))
        assert "Expected the number of predictions" in str(e.value)

    def test_munge_data(self, input_data, traces, rois, metadata):
        parser = InferenceParser(input_data=input_data, args=[])

        with open(parser.args["roi_masks_path"], "r") as f:
            raw_roi_data = json.load(f)

        roi_data = SparseAndDenseROISchema(many=True).load(raw_roi_data)
        roi_data, excluded_rois = filter_excluded_rois(roi_data)
        expected_rois = [coo_matrix(np.array(r["mask_matrix"])) for r in rois
                         if not r["exclusion_labels"]]
        actual_rois, actual_metadata, actual_traces, actual_np_traces = (
            _munge_data(parser, roi_data))
        for ix, e_roi in enumerate(expected_rois):
            np.testing.assert_array_equal(
                e_roi.toarray(), actual_rois[ix].toarray(),
                err_msg=("Expected ROIs did not equal actual ROIs from "
                         f" _munge_data for index={ix}."))
        np.testing.assert_array_equal(
            traces, actual_traces,
            err_msg=("Expected traces did not equal actual traces "
                     "from _munge_data"))
        np.testing.assert_array_equal(
            traces, actual_np_traces,
            err_msg=("Expected neuropil traces did not equal actual neuropil "
                     "traces from _munge_data"))
        assert [metadata]*len(expected_rois) == actual_metadata


class TestSparseAndDenseROISchema:
    def test_schema_makes_coos(self, rois):
        expected = [coo_matrix(np.array(r["mask_matrix"])) for r in rois]
        actual = SparseAndDenseROISchema(many=True).load(rois)
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
    def test_coo_roi_dump_single(
            self, data, expected_coo, additional_roi_json_data):
        """Cover the empty case, another unit test"""
        data.update(additional_roi_json_data)
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
    def test_coo_roi_dump_raise_error_mismatch_dimensions(
            self, data, additional_roi_json_data):
        with pytest.raises(ValidationError) as e:
            data.update(additional_roi_json_data)
            data.update({"exclusion_labels": [], "id": 42})
            SparseAndDenseROISchema().load(data)
        assert "Data in mask matrix did not correspond" in str(e.value)

    def test_schema_warns_empty(self, rois):
        rois[0]["height"] = 0
        rois[0]["width"] = 0
        with pytest.warns(UserWarning) as record:
            SparseAndDenseROISchema(many=True).load(rois)
        assert len(record) == 1
        assert "Input data contains empty ROI" in str(record[0].message)

    def test_schema_errors_shape_mismatch(self, rois):
        rois[0]["height"] = 100
        rois[0]["width"] = 29
        with pytest.raises(ValidationError) as e:
            SparseAndDenseROISchema(many=True).load(rois)
        assert "Data in mask matrix did not correspond" in str(e.value)
