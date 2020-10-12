import h5py
import numpy as np
from unittest.mock import patch, Mock
import pathlib
import argschema
import json
import ophys_etl.transforms.postprocess_rois as post_rois

import sys
sys.modules['suite2p'] = Mock()
import ophys_etl.transforms.suite2p_wrapper as s2pw  # noqa
import ophys_etl.pipelines.segment_postprocess_pipeline as sbpipe  # noqa


class MockSuite2PWrapper(argschema.ArgSchemaParser):
    default_schema = s2pw.Suite2PWrapperSchema
    default_output_schema = s2pw.Suite2PWrapperOutputSchema

    def run(self):
        stat = pathlib.Path(self.args['output_dir']) / "stat.npy"
        with open(stat, "w") as f:
            f.write("content")
        outj = {
                'output_files': {
                    'stat.npy': str(stat)
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


@patch(
        'ophys_etl.pipelines.segment_postprocess_pipeline.Suite2PWrapper',
        MockSuite2PWrapper)
@patch(
        'ophys_etl.pipelines.segment_postprocess_pipeline.PostProcessROIs',
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
            "convert_args": {
                "motion_correction_values": str(mcvalues_path)},
            "output_json": str(outj_path)
            }

    sbp = sbpipe.SegmentAndPostProcess(input_data=args, args=[])
    sbp.run()

    with open(outj_path, "r") as f:
        outj = json.load(f)

    assert outj == {'some_output': 'junk'}
