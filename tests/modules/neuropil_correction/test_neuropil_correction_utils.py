from pathlib import Path

import h5py
import numpy as np
import pytest

from ophys_etl.modules.neuropil_correction.utils import (
    NeuropilSubtract,
    estimate_contamination_ratios,
    fill_unconverged_r,
)

test_path = Path(__file__).parent.absolute() / "test_data"
args = {
    "neuropil_trace_file": str(test_path / "neuropil_traces.h5"),
    "roi_trace_file": str(test_path / "demix_traces.h5"),
}


@pytest.fixture()
def neuropil_traces():
    with h5py.File(args["neuropil_trace_file"], "r") as f:
        yield f["data"][:]


@pytest.fixture()
def roi_traces():
    with h5py.File(args["roi_trace_file"], "r") as f:
        yield f["data"][:]


def test_neuropil_subtract_fit__returns_correct_values(
    neuropil_traces, roi_traces
):
    F_N = neuropil_traces[1]
    F_M = roi_traces[1]

    ns = NeuropilSubtract()
    ns.set_F(F_M, F_N)
    ns.fit()

    expected_r_vals = np.concatenate(
        (
            np.arange(0, 2, 0.1),
            np.arange(0, 0.2, 0.01),
            np.arange(0.07, 0.091, 0.001),
        )
    )
    assert ns.r == pytest.approx(0.078, 1e-3)
    np.testing.assert_almost_equal(expected_r_vals, ns.r_vals)
    assert ns.error == pytest.approx(0.01887886, 1e-7)
    assert ns.error_vals[:3] == pytest.approx(
        [0.01888568174832354, 0.01887942348179789, 0.01889577822164021], 1e-7
    )


def test__estimate_contamination_ratios__returns_correct_dict(
    neuropil_traces, roi_traces
):

    results = estimate_contamination_ratios(roi_traces[1], neuropil_traces[1])

    expected_keys = ["r", "r_vals", "err", "err_vals", "min_error", "it"]
    assert list(results.keys()) == expected_keys


def test__fill_unconverged_r__returns_correct_values(
    neuropil_traces, roi_traces
):
    r_array = np.array([1.1, 0.078, 0.125])
    corrected_neuropil_traces = np.zeros_like(roi_traces)
    for i in range(len(r_array)):
        corrected_neuropil_traces[i] = (
            roi_traces[i] - r_array[i] * neuropil_traces[i]
        )

    (
        obt_corrected_neuropil_traces,
        obt_r_array,
        obt_rmse_array,
    ) = fill_unconverged_r(
        corrected_neuropil_traces, roi_traces, neuropil_traces, r_array
    )

    expected_corrected_traces_sliced = np.array(
        [
            [318.41422874, 259.60284182, 284.68170532],
            [84.88831718, 111.04645177, 115.9280698],
            [199.9796932, 233.80172784, 201.33909005],
        ]
    )
    expected_r_array = np.array([0.1015, 0.078, 0.125])
    expected_rmse_array = np.array([0.01268142, 0.01887886, 0.01142511])

    assert corrected_neuropil_traces.shape == roi_traces.shape
    np.testing.assert_array_almost_equal(
        expected_corrected_traces_sliced, obt_corrected_neuropil_traces[:, :3]
    )
    np.testing.assert_array_almost_equal(expected_r_array, obt_r_array)
    np.testing.assert_array_almost_equal(expected_rmse_array, obt_rmse_array)
