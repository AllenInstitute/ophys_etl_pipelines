import h5py
import numpy as np
from unittest.mock import patch, Mock
import pathlib
import argschema
import json
import pytest
import ophys_etl.modules.postprocess_rois.__main__ as post_rois

import sys
sys.modules['suite2p'] = Mock()
from ophys_etl.modules.suite2p_wrapper.schemas import \
        Suite2PWrapperSchema, Suite2PWrapperOutputSchema  # noqa: E402
import ophys_etl.modules.segment_postprocess.__main__ as sbpipe  # noqa


class MockSuite2PWrapper(argschema.ArgSchemaParser):
    default_schema = Suite2PWrapperSchema
    default_output_schema = Suite2PWrapperOutputSchema

    def run(self):
        stat = pathlib.Path(self.args['output_dir']) / "stat.npy"
        with open(stat, "w") as f:
            f.write("content")
        outj = {
                'output_files': {
                    'stat.npy': [str(stat)]
                    }
                }
        self.output(outj)


class MockOutputSchema(argschema.schemas.DefaultSchema):
    some_output = argschema.fields.Str(
        required=True)


class MockPostProcess(argschema.ArgSchemaParser):
    default_schema = post_rois.PostProcessROIsInputSchema
    default_output_schema = MockOutputSchema

    def run(self):
        self.output({'some_output': 'junk'})


@pytest.mark.suite2p_only
@patch(
        'ophys_etl.modules.segment_postprocess.__main__.Suite2PWrapper',
        MockSuite2PWrapper)
@patch(
        'ophys_etl.modules.segment_postprocess.__main__.PostProcessROIs',
        MockPostProcess)
def test_segment_postprocess_pipeline(tmp_path):
    """tests that satisfying the pipeline schema satisfies the
    internal transform schema.
    """
    h5path = tmp_path / "mc_video.h5"
    with h5py.File(str(h5path), "w") as f:
        f.create_dataset("data", data=np.zeros((20, 100, 100)))

    mcvalues_path = tmp_path / "mc_values.csv"
    with open(mcvalues_path, "w") as f:
        f.write("content")

    outj_path = tmp_path / "output.json"

    args = {"suite2p_args": {
                "h5py": str(h5path),
                "movie_frame_rate": 31.0,
            },
            "postprocess_args": {
                "motion_correction_values": str(mcvalues_path)},
            "output_json": str(outj_path)
            }

    sbp = sbpipe.SegmentAndPostProcess(input_data=args, args=[])
    sbp.run()

    with open(outj_path, "r") as f:
        outj = json.load(f)

    assert outj == {'some_output': 'junk'}
