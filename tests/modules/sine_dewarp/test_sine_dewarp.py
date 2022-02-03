import pytest
import h5py
import numpy as np
from pathlib import Path
import os

from ophys_etl.modules.sine_dewarp.__main__ import SineDewarp


@pytest.fixture
def input_h5(tmpdir):
    fname = tmpdir / "input.h5"
    with h5py.File(fname, "w") as f:
        f.create_dataset("data",
                         data=np.zeros((10, 512, 512), dtype="uint8"))
    yield str(fname)
    os.remove(fname)


def test_sine_dewarp_end_to_end(input_h5, tmpdir):
    raise RuntimeError("SFD says fail here")
    output_h5 = str(tmpdir / "output.h5")
    output_json = str(tmpdir / "output.json")
    args = {
            "input_h5": input_h5,
            "output_h5": output_h5,
            "equipment_name": "CAM2P.3",
            "aL": "160.0",
            "aR": "160.0",
            "bL": "85.0",
            "bR": "90.0",
            "chunk_size": 2,
            "output_json": output_json,
            "n_parallel_workers": 1}
    smod = SineDewarp(input_data=args, args=[])
    smod.run()

    assert Path(output_h5).is_file()
    assert Path(output_json).is_file()

    with h5py.File(output_h5, "r") as f:
        assert "data" in list(f.keys())
        assert f['data'].shape[0:2] == (10, 512)
        assert f['data'].shape[2] < 512
