import pytest
from unittest.mock import patch
from argschema import ArgSchemaParser
from marshmallow import ValidationError

from ophys_etl.modules.classifier_inference.__main__ import (
        InferenceInputSchema, main)


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
                    "trace_sampling_rate": 5,
                    "output_json": str(tmp_path / "output.json"),
                    "desired_trace_sampling_rate": 5},
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
                "trace_sampling_rate": 5,
                "desired_trace_sampling_rate": 5,
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

        with patch(
                "ophys_etl.modules.classifier_inference."
                "schemas.utils.RegistryConnection",
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
