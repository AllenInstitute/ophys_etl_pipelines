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
import ophys_etl.pipelines.suite2p_registration as s2preg  # noqa


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


@pytest.mark.suite2p_only
@pytest.mark.parametrize(
        ("excess_indices", "deltas", "thresh", "expected_indices"), [
            ([], [], 10, []),
            ([123, 567], [20, 20], 10, [123, 567]),
            ([123, 567], [8, 20], 10, [567]),
            ([123, 567], [-20, -20], 10, [123, 567]),
            ([123, 567], [-8, -20], 10, [567]),
            ([123, 567, 1234, 5678], [-20, -8, 8, 20], 10, [123, 5678])
            ])
def test_identify_and_clip_outliers(excess_indices, deltas,
                                    thresh, expected_indices):
    frame_index = np.arange(10000)
    # long-range drifts
    baseline = 20.0 * np.cos(2.0 * np.pi * frame_index / 500)
    additional = np.zeros_like(baseline)
    for index, delta in zip(excess_indices, deltas):
        additional[index] += delta

    data, indices = s2preg.identify_and_clip_outliers(
            baseline + additional,
            10,
            thresh)

    # check that the outliers were identified
    assert set(indices) == set(expected_indices)
    # check that the outlier values were clipped to within
    # the threshold of the underlying trend data
    # with a small delta value because the median-filtered data
    # is not quite ever the same as the baseline data
    deltas = np.abs(data[expected_indices] - baseline[expected_indices])
    small_delta = 1.0
    assert np.all(deltas < thresh + small_delta)


@pytest.mark.suite2p_only
@pytest.mark.parametrize("mock_ops_data", [
    {"Lx": 0, "Ly": 0, "nframes": 5, "xrange": 0, "yrange": 0,
     "xoff": [1, 2, 3, 4, 5], "yoff": [5, 4, 3, 2, 1],
     "corrXY": [6, 7, 8, 9, 10], "meanImg": 0}
])
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
        with patch('ophys_etl.pipelines.suite2p_registration.Suite2PWrapper',
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
