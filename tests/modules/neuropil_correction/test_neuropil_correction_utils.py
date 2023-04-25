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


@pytest.fixture()
def corrected_neuropil_traces(neuropil_traces, roi_traces, r_array):
    corrected_neuropil_traces = np.zeros_like(roi_traces)
    for i in range(len(r_array)):
        corrected_neuropil_traces[i] = (
            roi_traces[i] - r_array[i] * neuropil_traces[i]
        )
    return corrected_neuropil_traces


@pytest.fixture()
def r_array():
    return np.array([1.1, 0.078, 0.125])


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


@pytest.mark.parametrize(
    "repeat_number, expected_corrected_traces_sliced, expected_r_array, expected_rmse_array",  # noqa
    [
        (
            1,
            np.array(
                [
                    [236.62230614, 182.41965545, 204.852356],
                    [84.88831718, 111.04645177, 115.9280698],
                    [199.9796932, 233.80172784, 201.33909005],
                ]
            ),
            np.array([0.7, 0.078, 0.125]),
            np.array([0.01248130, 0.01887886, 0.01142511]),
        ),
        (
            3,
            np.repeat(
                np.array(
                    [
                        [318.41422874, 259.60284182, 284.68170532],
                        [84.88831718, 111.04645177, 115.9280698],
                        [199.9796932, 233.80172784, 201.33909005],
                    ]
                ),
                3,
                axis=0,
            ),
            np.repeat(np.array([0.1015, 0.078, 0.125]), 3),
            np.repeat(np.array([0.01268142, 0.01887886, 0.01142511]), 3),
        ),
    ],
)
def test__fill_unconverged_r__returns_correct_values(
    neuropil_traces,
    roi_traces,
    corrected_neuropil_traces,
    r_array,
    repeat_number,
    expected_corrected_traces_sliced,
    expected_r_array,
    expected_rmse_array,
):

    # Repeat dataset to test with dataset with more than 5 ROIs with r < 1
    # If dataset has less than 5 ROIs with r<1, values with r>1 are filled
    # with 0.7 instead of the mean
    neuropil_traces = np.repeat(neuropil_traces, repeat_number, axis=0)
    roi_traces = np.repeat(roi_traces, repeat_number, axis=0)
    corrected_neuropil_traces = np.repeat(
        corrected_neuropil_traces, repeat_number, axis=0
    )
    r_array = np.repeat(r_array, repeat_number)

    (
        obt_corrected_neuropil_traces,
        obt_r_array,
        obt_rmse_array,
    ) = fill_unconverged_r(
        corrected_neuropil_traces, roi_traces, neuropil_traces, r_array
    )

    assert corrected_neuropil_traces.shape == roi_traces.shape
    np.testing.assert_array_almost_equal(
        expected_corrected_traces_sliced, obt_corrected_neuropil_traces[:, :3]
    )
    np.testing.assert_array_almost_equal(expected_r_array, obt_r_array)
    np.testing.assert_array_almost_equal(expected_rmse_array, obt_rmse_array)
