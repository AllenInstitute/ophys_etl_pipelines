import numpy as np
import h5py
import pytest
import json

from ophys_etl.pipelines.dff import DffJob, DffJobSchema
from ophys_etl.pipelines import dff


@pytest.fixture
def trace_h5(tmp_path):
    with h5py.File(tmp_path / "input_trace.h5", "w") as f:
        f.create_dataset("FC", data=np.ones((3, 100)))
        f.create_dataset("roi_names",
                         data=np.array([b'abc', b'123', b'drm'], dtype='|S3'))
    f.close()
    yield


def test_dff_job_run(tmp_path, trace_h5, monkeypatch):
    def mock_dff(x, y, z): return x, 0.1, 10

    args = {
        "input_file": str(tmp_path / "input_trace.h5"),
        "output_file": str(tmp_path / "output_dff.h5"),
        "output_json": str(tmp_path / "output_json.json"),
        "movie_frame_rate_hz": 10.0,
    }
    dff_job = DffJob(input_data=args, args=[])
    monkeypatch.setattr(dff, "compute_dff_trace", mock_dff)
    dff_job.run()
    # Load the output and check
    with h5py.File(args["output_file"], "r") as f:
        assert list(f.keys()) == [
            "data", "num_small_baseline_frames", "roi_names", "sigma_dff"]
        np.testing.assert_array_equal(np.ones((3, 100)), f["data"][()])
        np.testing.assert_array_equal(
            np.array([b'abc', b'123', b'drm'], dtype='|S3'),
            f["roi_names"][()])
        np.testing.assert_allclose(     # float stuff...
            np.array([0.1, 0.1, 0.1]), f["sigma_dff"])
        np.testing.assert_array_equal(
            np.array([10, 10, 10]),
            f["num_small_baseline_frames"])
    with open(args["output_json"], "r") as f:
        assert {} == json.load(f)


@pytest.mark.parametrize(
    "frame_rate, short_filter_s, long_filter_s, expected_short, expected_long",
    [
        (30., 3.3, 600., 99, 18001),
        (11.1, 10.0, 1001., 111, 11111)
    ]
)
def test_dff_schema_post_load(tmp_path, trace_h5, frame_rate, short_filter_s,
                              long_filter_s, expected_short, expected_long):
    args = {
        "input_file": str(tmp_path / "input_trace.h5"),
        "output_file": str(tmp_path / "output_dff.h5"),
        "long_baseline_filter_s": long_filter_s,
        "short_filter_s": short_filter_s,
        "movie_frame_rate_hz": frame_rate
    }
    data = DffJobSchema().load(args)
    assert data["short_filter_frames"] == expected_short
    assert isinstance(data["short_filter_frames"], int)
    assert data["long_filter_frames"] == expected_long
    assert isinstance(data["long_filter_frames"], int)
