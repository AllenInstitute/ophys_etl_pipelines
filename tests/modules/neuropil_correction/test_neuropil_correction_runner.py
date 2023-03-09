import json
import os
from pathlib import Path

import h5py
import numpy as np

from ophys_etl.modules.neuropil_correction.__main__ import (
    NeuropilCorrectionRunner,
)


def test_neuropil_correction_runner(tmp_path):

    test_path = Path(__file__).parent.absolute() / "test_data"
    args = {
        "neuropil_trace_file": str(test_path / "neuropil_traces.h5"),
        "roi_trace_file": str(test_path / "demix_traces.h5"),
        "storage_directory": str(tmp_path),
        "motion_corrected_stack": str(test_path / "demix_traces.h5"),
        "output_json": os.path.join(tmp_path, "output.json"),
    }
    expected_output_json = {k: args[k] for k in list(args.keys())[:-1]}
    expected_output_json["neuropil_correction"] = str(
        tmp_path / "neuropil_correction.h5"
    )

    neuropil_correction = NeuropilCorrectionRunner(input_data=args, args=[])
    neuropil_correction.run()

    with open(args["output_json"]) as f:
        args_out = json.load(f)

    assert expected_output_json == {
        k: v for k, v in args_out.items() if k in expected_output_json
    }

    with h5py.File(args_out["neuropil_correction"], "r") as f:
        np.testing.assert_array_almost_equal(
            np.array(
                [
                    [236.62230614, 182.41965545, 204.852356],
                    [84.88831718, 111.04645177, 115.9280698],
                    [199.9796932, 233.80172784, 201.33909005],
                ]
            ),
            f["FC"][:, :3],
        )
        np.testing.assert_array_almost_equal(
            np.array([0.7, 0.078, 0.125]), f["r"][:]
        )
        np.testing.assert_array_equal(
            np.array([b"1109811483", b"1109811486", b"1109811490"]),
            f["roi_names"][:],
        )
        np.testing.assert_array_almost_equal(
            np.array([0.01248130, 0.01887886, 0.01142511]), f["RMSE"][:]
        )
