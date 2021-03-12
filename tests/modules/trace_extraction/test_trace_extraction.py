import h5py
import json
import numpy as np
from pathlib import Path

import ophys_etl.modules.trace_extraction.__main__ as temod


def test_TraceExtraction(tmpdir, monkeypatch):
    motion_stack = tmpdir / "input.h5"
    with h5py.File(motion_stack, "w") as f:
        f.create_dataset("data", data=np.zeros((10, 10, 10), dtype='uint8'))
    log0 = tmpdir / "log.txt"
    with open(log0, "w") as f:
        f.write("stuff")
    outj = tmpdir / "output.json"
    args = {
            "motion_border": {
                "x0": 0,
                "x1": 0,
                "y0": 0,
                "y1": 0},
            "storage_directory": str(tmpdir),
            "motion_corrected_stack": str(motion_stack),
            "rois": [{
                "mask": [[True]],
                "x": 0,
                "y": 0,
                "width": 1,
                "height": 1,
                "valid": True,
                "id": 123456}],
            "log_0": str(log0),
            "output_json": str(outj)}

    def mock_extract_traces(a, b, c, d):
        tfile = tmpdir / "traces.h5"
        with h5py.File(tfile, "w") as f:
            f.create_dataset("data", data=0)
        npfile = tmpdir / "nptraces.h5"
        with h5py.File(npfile, "w") as f:
            f.create_dataset("data", data=0)
        val = {
                "neuropil_trace_file": str(npfile),
                "roi_trace_file": str(tfile),
                "exclusion_labels": []}
        return val
    monkeypatch.setattr(temod, "extract_traces", mock_extract_traces)
    te = temod.TraceExtraction(input_data=args, args=[])
    te.run()

    assert Path(outj).is_file()
    with open(outj, "r") as f:
        j = json.load(f)
    assert Path(j['roi_trace_file']).is_file()
    assert Path(j['neuropil_trace_file']).is_file()
