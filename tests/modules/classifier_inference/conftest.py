import pytest
import boto3
import json
import h5py
import numpy as np
from moto import mock_s3


# ROI input data not used but needed to pass through to the module output
# Define here and reuse for brevity
@pytest.fixture
def additional_roi_json_data():
    return {
            "max_correction_up": 0,
            "max_correction_down": 0,
            "max_correction_left": 0,
            "max_correction_right": 0,
            "valid_roi": True,
            "mask_image_plane": 0}


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
    client.create_bucket(Bucket="mybucket",
                         CreateBucketConfiguration={
                             'LocationConstraint': 'us-west-2'})
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
def rois(additional_roi_json_data):
    return [
        {
            "id": 0,
            "x": 1,
            "y": 3,
            "height": 3,
            "width": 2,
            "mask_matrix": [[False, True], [True, True], [True, True]],
            "exclusion_labels": [],
            **additional_roi_json_data
        },
        {
            "id": 1,
            "x": 1,
            "y": 3,
            "height": 1,
            "width": 4,
            "mask_matrix": [[True, False, False, True]],
            "exclusion_labels": [],
            **additional_roi_json_data
        },
        {
            "id": 2,
            "x": 9,
            "y": 2,
            "height": 2,
            "width": 2,
            "mask_matrix": [[True, False], [True, True]],
            "exclusion_labels": [],
            **additional_roi_json_data
        },
        {
            "id": 3,
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
    trace_f["roi_names"] = np.array(['0', '1', '2']).astype(np.string_)
    np_trace_f = h5py.File(np_trace_path, "w")
    np_trace_f["data"] = traces
    np_trace_f["roi_names"] = np.array(['0', '1', '2']).astype(np.string_)
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
        "trace_sampling_rate": 1,
        "desired_trace_sampling_rate": 1,
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
