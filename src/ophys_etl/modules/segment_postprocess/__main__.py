import json
import argschema

from ophys_etl.modules.postprocess_rois.__main__ import PostProcessROIs
from ophys_etl.modules.suite2p_wrapper.__main__ import Suite2PWrapper
from ophys_etl.modules.segment_postprocess.schemas import \
        SegmentPostProcessSchema


class SegmentAndPostProcess(argschema.ArgSchemaParser):
    default_schema = SegmentPostProcessSchema

    def run(self):

        # segment with Suite2P, almost all default args
        suite2p_args = self.args['suite2p_args']
        segment = Suite2PWrapper(input_data=suite2p_args, args=[])
        segment.run()

        # postprocess and output LIMS format
        with open(suite2p_args["output_json"], "r") as f:
            stat_path = json.load(f)['output_files']['stat.npy'][0]

        postprocess_args = self.args['postprocess_args']
        if "suite2p_stat_path" not in postprocess_args:
            postprocess_args["suite2p_stat_path"] = stat_path

        postprocess = PostProcessROIs(input_data=postprocess_args, args=[])
        postprocess.run()

        # Clean up temporary directories and/or files created during
        # Schema invocation
        if self.schema.tmpdir is not None:
            self.schema.tmpdir.cleanup()


if __name__ == "__main__":  # pragma: nocover
    sbmod = SegmentAndPostProcess()
    sbmod.run()
