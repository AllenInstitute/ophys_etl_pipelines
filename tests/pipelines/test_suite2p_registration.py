import numpy as np
from pathlib import Path
import h5py
import tifffile
import argschema
import pytest
import json
from unittest.mock import patch, Mock
import sys
sys.modules['suite2p'] = Mock()
import ophys_etl.transforms.suite2p_wrapper as s2pw  # noqa
import ophys_etl.pipelines.suite2p_registration as s2preg  # noqa


class MockSuite2PWrapper(argschema.ArgSchemaParser):
    default_schema = s2pw.Suite2PWrapperSchema
    default_output_schema = s2pw.Suite2PWrapperOutputSchema

    def run(self):
        fname_base = Path(self.args['output_dir'])
        fnames = []
        for i in range(10):
            data = np.ones((10, 32, 32), 'int16') * i
            fnames.append(fname_base / f"my_tiff_{i}.tif")
            tifffile.imwrite(fnames[-1], data)

        ops_path = Path(self.args['output_dir']) / "ops.npy"
        ops_keys = ["Lx", "Ly", "nframes", "xrange", "yrange", "xoff", "yoff",
                    "corrXY", "meanImg"]
        ops_dict = {k: 0 for k in ops_keys}
        np.save(ops_path, ops_dict)
        outj = {
                'output_files': {
                    'ops.npy': [str(ops_path)],
                    '*.tif': [str(i) for i in fnames]
                    }
                }
        self.output(outj)


@pytest.mark.suite2p_only
@patch(
        'ophys_etl.pipelines.suite2p_registration.Suite2PWrapper',
        MockSuite2PWrapper)
def test_suite2p_registration(tmp_path):
    h5path = tmp_path / "mc_video.h5"
    with h5py.File(str(h5path), "w") as f:
        f.create_dataset("data", data=np.zeros((20, 100, 100)))

    outj_path = tmp_path / "output.json"
    args = {"suite2p_args": {
                "h5py": str(h5path),
            },
            "motion_corrected_output": str(tmp_path / "motion_output.h5"),
            "motion_diagnostics_output":
                str(tmp_path / "motion_diagnostics.h5"),
            "output_json": str(outj_path)}

    reg = s2preg.Suite2PRegistration(input_data=args, args=[])
    reg.run()

    with open(outj_path, "r") as f:
        outj = json.load(f)

    for k in ['motion_corrected_output', 'motion_diagnostics_output']:
        assert k in outj
        with h5py.File(outj[k], "r") as f:
            pass

    with h5py.File(outj['motion_corrected_output'], "r") as f:
        data = f['data'][()]
    assert data.shape == (100, 32, 32)
    assert data.max() == 9
    assert data.min() == 0

    with h5py.File(outj['motion_diagnostics_output'], "r") as f:
        keys = list(f.keys())
        for k in keys:
            assert f[k][()] == 0
