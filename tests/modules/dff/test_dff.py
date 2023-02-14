import numpy as np
import h5py
import json
import time
import pytest
import ophys_etl.modules.dff.__main__ as dff_main
from ophys_etl.modules.dff.__main__ import DffJob


def test_dff_job_run(tmp_path, trace_h5, monkeypatch):
    def mock_dff(x, y, z): return x, 0.1, 10
    monkeypatch.setattr(dff_main, "compute_dff_trace", mock_dff)
    monkeypatch.setattr(time, "time", lambda: 123456789)

    args = {
        "input_file": str(tmp_path / "input_trace.h5"),
        "output_file": str(tmp_path / "output_dff.h5"),
        "output_json": str(tmp_path / "output_json.json"),
        "movie_frame_rate_hz": 10.0,
    }
    expected_output = {"output_file": args["output_file"],
                       "created_at": 123456789}

    dff_job = DffJob(input_data=args, args=[])
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
        # Argschema automatically puts some default info into the output,
        # so just testing for what we care about in case argschema changes
        actual_output = json.load(f)
        assert expected_output == {
            k: v for k, v in actual_output.items() if k in expected_output}


def test_dff_trace(monkeypatch):
    """
    Notes:
    If we don't constrain this it's very unwieldy. Not using
    parametrization because these values need to be
    monkeypatched thoughtfully to make a unit test work out
    Isn't a great candidate for mock because most of the
    logic pertains to filtering numpy arrays anyway.
    """
    monkeypatch.setattr(dff_main, "noise_std", lambda x, y: 1.0)
    monkeypatch.setattr(dff_main, "nanmedian_filter", lambda x, y: x-1.0)
    f_trace = np.array([1.1, 2., 3., 3., 3., 11.])    # 2 "small baseline"

    dff, sigma, small_baseline = dff_main.compute_dff_trace(f_trace, 1, 1)
    assert 2 == small_baseline
    assert 1.0 == sigma     # monkeypatched noise_std
    expected = np.array([1, 1, 0.5, 0.5, 0.5, 0.1])
    np.testing.assert_array_equal(expected, dff)

    with pytest.raises(ValueError):
        dff_main.compute_dff_trace(
            f_trace, long_filter_length=2, short_filter_length=1)
    with pytest.raises(ValueError):
        dff_main.compute_dff_trace(
            f_trace, long_filter_length=3, short_filter_length=2)
    with pytest.raises(ValueError):
        dff_main.compute_dff_trace(
            f_trace, long_filter_length=7, short_filter_length=1)
