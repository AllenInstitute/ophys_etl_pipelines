import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import argschema
import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile

sys.modules['suite2p'] = Mock()
sys.modules['suite2p.registration.rigid'] = Mock()
from ophys_etl.modules.suite2p_wrapper.schemas import \
        Suite2PWrapperSchema, Suite2PWrapperOutputSchema  # noqa: E402
import ophys_etl.modules.suite2p_registration.__main__ as s2preg  # noqa


class MockSuite2PWrapper(argschema.ArgSchemaParser):
    default_schema = Suite2PWrapperSchema
    default_output_schema = Suite2PWrapperOutputSchema
    mock_ops_data = None

    def run(self):
        fname_base = Path(self.args['output_dir'])
        fnames = []
        for i in range(10):
            data = np.ones((100, 32, 32), 'int16') * i
            fnames.append(fname_base / f"my_tiff_{i}.tif")
            tifffile.imwrite(fnames[-1], data)

        ops_path = Path(self.args['output_dir']) / "ops.npy"
        ops_keys = ["Lx", "Ly", "nframes", "xrange", "yrange", "xoff", "yoff",
                    "corrXY", "meanImg"]

        if self.mock_ops_data is None:
            ops_dict = {k: 0 for k in ops_keys}
        else:
            ops_dict = self.mock_ops_data
        self.logger.info(f"Saving ops_dict with: {ops_dict}")

        np.save(ops_path, ops_dict)
        outj = {
                'output_files': {
                    'ops.npy': [str(ops_path)],
                    '*.tif': [str(i) for i in fnames]
                    }
                }
        self.output(outj)


@pytest.fixture
def mock_ops_data():
    return {"Lx": 0, "Ly": 0, "nframes": 5, "xrange": 0, "yrange": 0,
            "xoff": [1, 2, 3, 4, 5], "yoff": [5, 4, 3, 2, 1],
            "corrXY": [6, 7, 8, 9, 10], "meanImg": 0}


def test_suite2p_registration(tmp_path, mock_ops_data):
    h5path = tmp_path / "mc_video.h5"
    with h5py.File(str(h5path), "w") as f:
        f.create_dataset("data", data=np.zeros((1000, 32, 32)))

    outj_path = tmp_path / "output.json"
    args = {"suite2p_args": {
                "h5py": str(h5path),
            },
            'movie_frame_rate_hz': 11.0,
            'registration_summary_output': str(tmp_path / "summary.png"),
            'motion_correction_preview_output': str(tmp_path / "preview.webm"),
            "motion_corrected_output": str(tmp_path / "motion_output.h5"),
            "motion_diagnostics_output": str(tmp_path / "motion_offset.csv"),
            "max_projection_output": str(tmp_path / "max_proj.png"),
            "avg_projection_output": str(tmp_path / "avg_proj.png"),
            "output_json": str(outj_path)}

    with patch.object(MockSuite2PWrapper, "mock_ops_data", mock_ops_data):
        with patch(
            'ophys_etl.modules.suite2p_registration.__main__.Suite2PWrapper',
                MockSuite2PWrapper):
            reg = s2preg.Suite2PRegistration(input_data=args, args=[])
            reg.run()

    # Test that output files exist
    for k in ['registration_summary_output',
              'motion_correction_preview_output',
              'motion_corrected_output',
              'motion_diagnostics_output',
              'max_projection_output',
              'avg_projection_output',
              'output_json']:
        assert Path(reg.args[k]).exists

    with open(outj_path, "r") as f:
        outj = json.load(f)

    # Test that motion_corrected_output field exists and
    # that h5 file contains correct data
    assert 'motion_corrected_output' in outj
    with h5py.File(outj['motion_corrected_output'], "r") as f:
        data = f['data'][()]
    assert data.shape == (1000, 32, 32)
    assert data.max() == 9
    assert data.min() == 0

    # Test that motion_diagnostics_output field exists and that csv
    # file contains correct data
    assert 'motion_diagnostics_output' in outj
    obt_motion_offset_df = pd.read_csv(outj['motion_diagnostics_output'])
    expected_frame_indices = list(range(mock_ops_data['nframes']))
    assert np.allclose(obt_motion_offset_df["framenumber"],
                       expected_frame_indices)
    assert np.allclose(obt_motion_offset_df["x"], mock_ops_data['xoff'])
    assert np.allclose(obt_motion_offset_df["y"], mock_ops_data['yoff'])
    assert np.allclose(obt_motion_offset_df["correlation"],
                       mock_ops_data['corrXY'])


def test_suite2p_frame_rate_consistency(tmp_path, mock_ops_data):
    """
    Test that, if suite2p_args and args specify different
    movie_frame_rate_hz, an exception is raised
    """
    h5path = tmp_path / "mc_video.h5"
    with h5py.File(str(h5path), "w") as f:
        f.create_dataset("data", data=np.zeros((1000, 32, 32)))

    outj_path = tmp_path / "output.json"
    args = {"suite2p_args": {
                "h5py": str(h5path),
                "movie_frame_rate_hz": 7.0
            },
            'movie_frame_rate_hz': 11.0,
            'registration_summary_output': str(tmp_path / "summary.png"),
            'motion_correction_preview_output': str(tmp_path / "preview.webm"),
            "motion_corrected_output": str(tmp_path / "motion_output.h5"),
            "motion_diagnostics_output": str(tmp_path / "motion_offset.csv"),
            "max_projection_output": str(tmp_path / "max_proj.png"),
            "avg_projection_output": str(tmp_path / "avg_proj.png"),
            "output_json": str(outj_path)}

    with pytest.raises(ValueError, match="two values of movie_frame_rate_hz"):
        with patch.object(MockSuite2PWrapper, "mock_ops_data", mock_ops_data):
            with patch(
              'ophys_etl.modules.suite2p_registration.__main__.Suite2PWrapper',
              MockSuite2PWrapper):
                s2preg.Suite2PRegistration(input_data=args, args=[])

    # now check that movie_frame_rate_hz gets propagated into suite2p_args
    # correctly
    args = {"suite2p_args": {
                "h5py": str(h5path),
            },
            'movie_frame_rate_hz': 11.0,
            'registration_summary_output': str(tmp_path / "summary.png"),
            'motion_correction_preview_output': str(tmp_path / "preview.webm"),
            "motion_corrected_output": str(tmp_path / "motion_output.h5"),
            "motion_diagnostics_output": str(tmp_path / "motion_offset.csv"),
            "max_projection_output": str(tmp_path / "max_proj.png"),
            "avg_projection_output": str(tmp_path / "avg_proj.png"),
            "output_json": str(outj_path)}

    with patch.object(MockSuite2PWrapper, "mock_ops_data", mock_ops_data):
        with patch(
            'ophys_etl.modules.suite2p_registration.__main__.Suite2PWrapper',
                MockSuite2PWrapper):
            runner = s2preg.Suite2PRegistration(input_data=args, args=[])
            assert runner.args['suite2p_args']['movie_frame_rate_hz'] == 11.0
