import json
import math
import os.path
from unittest.mock import patch

import boto3
import h5py
import joblib
import numpy as np
import pytest
from argschema import ArgSchemaParser
from marshmallow import ValidationError
from moto import mock_s3
from scipy.sparse import coo_matrix
from ophys_etl.transforms.classification import (InferenceInputSchema,
                                                 InferenceParser,
                                                 SparseAndDenseROISchema,
                                                 _munge_data, downsample,
                                                 load_model, main,
                                                 filtered_roi_load)
# ROI input data not used but needed to pass through to the module output
# Define here and reuse for brevity
additional_roi_json_data = {
    "max_correction_up": 0,
    "max_correction_down": 0,
    "max_correction_left": 0,
    "max_correction_right": 0,
    "valid_roi": True,
    "id": 42,
    "mask_image_plane": 0
}


@pytest.fixture(scope="function")
def mock_model():
    def _create_mock_model(predict_fn):
        class MockModel():
            def __init__(self):
                pass

            def predict(self, x):
                return predict_fn(x)
        return MockModel()
    return _create_mock_model


@pytest.fixture(scope="function")
def classifier_model(tmp_path):
    with open(tmp_path / "hello.txt", "w") as f:
        f.write("abc")
    yield str(tmp_path / "hello.txt")


@pytest.fixture(scope="function")
def traces():
    return [np.arange(100), np.arange(100), np.arange(100)]


@pytest.fixture(scope="function")
def s3_classifier(classifier_model):
    s3 = mock_s3()
    s3.start()
    client = boto3.client("s3")
    client.create_bucket(Bucket="mybucket")
    client.upload_file(classifier_model, "mybucket", "hello.txt")
    yield "s3://mybucket/hello.txt"
    s3.stop()


@pytest.fixture(scope="function")
def metadata():
    return {
        "rig": "rig",
        "depth": 100,
        "full_genotype": "Vip",
        "targeted_structure": "nAcc",
    }


@pytest.fixture(scope="function")
def rois():
    return [
        {
            "x": 1,
            "y": 3,
            "height": 3,
            "width": 2,
            "mask_matrix": [[False, True], [True, True], [True, True]],
            "exclusion_labels": [],
            **additional_roi_json_data
        },
        {
            "x": 1,
            "y": 3,
            "height": 1,
            "width": 4,
            "mask_matrix": [[True, False, False, True]],
            "exclusion_labels": [],
            **additional_roi_json_data
        },
        {
            "x": 9,
            "y": 2,
            "height": 2,
            "width": 2,
            "mask_matrix": [[True, False], [True, True]],
            "exclusion_labels": [],
            **additional_roi_json_data
        },
        {
            "x": 9,
            "y": 2,
            "height": 2,
            "width": 2,
            "mask_matrix": [[True, False], [True, True]],
            "exclusion_labels": ["motion_border"],
            **additional_roi_json_data
        }
    ]


@pytest.fixture(scope="function")
def input_data(tmp_path, classifier_model, traces, rois, metadata):
    trace_path = str(tmp_path / "trace.h5")
    np_trace_path = str(tmp_path / "np_trace.h5")
    trace_f = h5py.File(trace_path, "w")
    trace_f["data"] = traces
    np_trace_f = h5py.File(np_trace_path, "w")
    np_trace_f["data"] = traces
    trace_f.close()
    np_trace_f.close()

    roi_masks_path = str(tmp_path / "rois.json")

    with open(roi_masks_path, "w") as f:
        json.dump(rois, f)

    schema_input = {
        "neuropil_traces_path": np_trace_path,
        "traces_path": trace_path,
        "roi_masks_path": roi_masks_path,
        "classifier_model_path": classifier_model,
        "trace_sampling_fps": 1,
        "downsample_to": 1,
        "output_json": str(tmp_path / "output.json"),
        **metadata
    }
    yield schema_input


@pytest.fixture(scope="function")
def h5_file(tmp_path):
    fp = tmp_path / "test.h5"
    with h5py.File(tmp_path / "test.h5", "w") as f:
        f.create_dataset("data", (10,), dtype="i")
        f["data"] == range(10)
    return (str(fp), "data")


@pytest.fixture(scope="function")
def good_roi_json(tmp_path):
    fp = tmp_path / "masks.json"
    data = [
        {
            'x': 1,
            'y': 2,
            'width': 2,
            'height': 2,
            'mask': [[False, False], [True, False]],
            'exclusion_labels': [],
        }
    ]
    with open(fp, "w") as f:
        json.dump(data, f)
    return str(fp)


@pytest.fixture(scope="function")
def bad_roi_json(tmp_path):
    fp = tmp_path / "masks.json"
    data = [
        {
            'x': 1,
            'y': 2,
            'width': "a",
            'height': 2,
            'valid_roi': True,
            'exclusion_labels': [],
        }
    ]
    with open(fp, "w") as f:
        json.dump(data, f)
    return str(fp)


class TestInferenceInputSchema:

    @pytest.mark.parametrize(
        "np_key, traces_key, error_text", [
            (
                "bale", "data",
                "(neuropil_traces_data_key) was missing in h5 file",
            ),
            (
                "data", "howl",
                "(traces_data_key) was missing in h5 file",
            ),
        ]
    )
    def test_fails_wrong_h5_key(
            self, np_key, traces_key, error_text, h5_file, good_roi_json,
            classifier_model, tmp_path):
        with pytest.raises(ValidationError) as e:
            ArgSchemaParser(
                input_data={
                    "neuropil_traces_path": h5_file[0],
                    "neuropil_traces_data_key": np_key,
                    "traces_path": h5_file[0], "traces_data_key": traces_key,
                    "roi_masks_path": good_roi_json,
                    "rig": "rig",
                    "targeted_structure": "nAcc",
                    "depth": 100,
                    "classifier_model_path": classifier_model,
                    "full_genotype": "vip",
                    "trace_sampling_fps": 5,
                    "output_json": str(tmp_path / "output.json"),
                    "downsample_to": 5},
                schema_type=InferenceInputSchema,
                args=[])
        assert error_text in str(e.value)

    @pytest.mark.parametrize(
        "np_key, traces_key", [
            ("data", "data"),
        ]
    )
    def test_succeeds_correct_h5_keys(
            self, np_key, traces_key, h5_file, good_roi_json, classifier_model,
            tmp_path):
        ArgSchemaParser(
            input_data={
                "neuropil_traces_path": h5_file[0],
                "neuropil_traces_data_key": np_key,
                "traces_path": h5_file[0],
                "traces_data_key": traces_key,
                "roi_masks_path": good_roi_json,
                "rig": "rig",
                "targeted_structure": "nAcc",
                "depth": 100,
                "classifier_model_path": classifier_model,
                "full_genotype": "vip",
                "trace_sampling_fps": 5,
                "downsample_to": 5,
                "output_json": str(tmp_path / "output.json")
                },
            schema_type=InferenceInputSchema,
            args=[])

    def test_classifier_model_path(
            self, classifier_model, s3_classifier, input_data, tmp_path):
        ArgSchemaParser(
            input_data=input_data,    # input_data has local classifier default
            schema_type=InferenceInputSchema,
            args=[])
        s3_input = input_data.copy()
        s3_input["classifier_model_path"] = s3_classifier
        ArgSchemaParser(
            input_data=s3_input,
            schema_type=InferenceInputSchema,
            args=[])
        fake_s3 = input_data.copy()
        fake_s3["classifier_model_path"] = "s3://my-bucket/fake-hello.txt"
        with pytest.raises(ValidationError) as e:
            ArgSchemaParser(
                input_data=fake_s3,
                schema_type=InferenceInputSchema,
                args=[])
        assert "does not exist" in str(e.value)
        fake_local = input_data.copy()
        fake_local["classifier_model_path"] = str(tmp_path / "hello-again.txt")
        with pytest.raises(ValidationError) as e:
            ArgSchemaParser(
                input_data=fake_local,
                schema_type=InferenceInputSchema,
                args=[])
        assert "does not exist" in str(e.value)

    def test_invalid_model_registry_env(self, input_data):
        # This should run without error
        ArgSchemaParser(input_data=input_data,
                        schema_type=InferenceInputSchema,
                        args=[])
        # Now to test when the model_registry_env field is invalid
        invalid_model_registry_env = input_data.copy()
        invalid_model_registry_env["model_registry_env"] = "prode"
        with pytest.raises(ValidationError) as e:
            ArgSchemaParser(input_data=invalid_model_registry_env,
                            schema_type=InferenceInputSchema,
                            args=[])
        assert "not a valid value for the 'model_registry_env'" in str(e.value)

    @pytest.mark.parametrize("patch_input_data", [
        ({}),
        ({"model_registry_table_name": "Test1", "model_registry_env": "dev"}),
        ({"model_registry_table_name": "Test2", "model_registry_env": "stage"})
    ])
    def test_determine_classifier_model_path(self, input_data,
                                             classifier_model,
                                             patch_input_data):
        no_classifier_model_path = input_data.copy()
        no_classifier_model_path.pop("classifier_model_path", None)
        no_classifier_model_path.update(patch_input_data)

        table_name = patch_input_data.get("model_registry_table_name",
                                          "ROIClassifierRegistry")
        env = patch_input_data.get("model_registry_env", "prod")

        with patch("ophys_etl.transforms.classification.RegistryConnection",
                   autospec=True) as MockRegistryConnection:
            connection = MockRegistryConnection.return_value
            connection.get_active_model.return_value = classifier_model

            parser = ArgSchemaParser(input_data=no_classifier_model_path,
                                     schema_type=InferenceInputSchema,
                                     args=[])
            args = parser.args

            assert args["classifier_model_path"] == classifier_model
            MockRegistryConnection.assert_called_with(table_name=table_name)
            connection.get_active_model.assert_called_with(env=env)

    def test_fails_invalid_roi(
            self, input_data, bad_roi_json):
        input_data.update({"roi_masks_path": bad_roi_json})
        with pytest.raises(ValidationError) as e:
            parser = ArgSchemaParser(
                input_data=input_data,
                schema_type=InferenceInputSchema,
                args=[])
            main(parser)
        assert "Missing data for required field" in str(e.value)
        assert "Not a valid integer" in str(e.value)


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
            client.create_bucket(Bucket="mybucket")
            client.upload_file(model_path, "mybucket", "my_model.joblib")
            uri = "s3://mybucket/my_model.joblib"
            obt = load_model(uri)
    else:
        uri = model_path
        obt = load_model(uri)
    assert obt == model_data


def test_filtered_roi_loader(input_data):
    parser = InferenceParser(input_data=input_data, args=[])
    roi_data, excluded = filtered_roi_load(parser.args["roi_masks_path"])
    assert len(roi_data) == 3
    assert len(excluded) == 1


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
        roi_data, excluded = filtered_roi_load(parser.args["roi_masks_path"])
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
    def test_coo_roi_dump_single(self, data, expected_coo):
        """Cover the empty case, another unit test"""
        data.update(additional_roi_json_data)
        data.update({"exclusion_labels": []})
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
    def test_coo_roi_dump_raise_error_mismatch_dimensions(self, data):
        with pytest.raises(ValidationError) as e:
            data.update(additional_roi_json_data)
            data.update({"exclusion_labels": []})
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
