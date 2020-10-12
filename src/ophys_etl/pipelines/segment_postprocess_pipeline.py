import json
from pathlib import Path
import tempfile

import marshmallow
import argschema


from ophys_etl.transforms.postprocess_rois import (
    PostProcessROIsInputSchema, PostProcessROIs)
from ophys_etl.transforms.suite2p_wrapper import (Suite2PWrapper,
                                                  Suite2PWrapperSchema)


class ConvertROIsInputSchema(PostProcessROIsInputSchema):

    # Override the default "suite2p_stat_field" required parameter of the
    # PostProcessROIsInputSchema.
    # When run in pipeline, the suite2p_stat_path won't be known until
    # after segmentation has run.
    suite2p_stat_path = argschema.fields.Str(
        required=False,
        validate=lambda x: Path(x).exists(),
        description=("Path to s2p output stat file containing ROIs generated "
                     "during source extraction"))


class SegmentPostProcessSchema(argschema.ArgSchema):
    suite2p_args = argschema.fields.Nested(Suite2PWrapperSchema,
                                           required=True)
    convert_args = argschema.fields.Nested(ConvertROIsInputSchema,
                                           required=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmpdir = None

    @marshmallow.pre_load
    def setup_default_suite2p_args(self, data: dict, **kwargs) -> dict:

        # suite2p_args
        if "output_dir" not in data["suite2p_args"]:
            self.tmpdir = tempfile.TemporaryDirectory()
            data["suite2p_args"]["output_dir"] = self.tmpdir.name

        if "output_json" not in data["suite2p_args"]:
            Suite2p_output = (Path(data["suite2p_args"]["output_dir"])
                              / "Suite2P_output.json")
            data["suite2p_args"]["output_json"] = str(Suite2p_output)

        data["suite2p_args"]["log_level"] = data["log_level"]

        # convert_args
        if "motion_corrected_video" not in data["convert_args"]:
            data["convert_args"]["motion_corrected_video"] = (
                data["suite2p_args"]["h5py"])

        if "output_json" not in data["convert_args"]:
            data["convert_args"]["output_json"] = data["output_json"]

        data["convert_args"]["log_level"] = data["log_level"]

        return data


class SegmentAndPostProcess(argschema.ArgSchemaParser):
    default_schema = SegmentPostProcessSchema

    def run(self):

        # segment with Suite2P, almost all default args
        suite2p_args = self.args['suite2p_args']
        segment = Suite2PWrapper(input_data=suite2p_args, args=[])
        segment.run()

        # binarize and output LIMS format
        with open(suite2p_args["output_json"], "r") as f:
            stat_path = json.load(f)['output_files']['stat.npy']

        convert_args = self.args['convert_args']
        if "suite2p_stat_path" not in convert_args:
            convert_args["suite2p_stat_path"] = stat_path

        postprocess = PostProcessROIs(input_data=convert_args, args=[])
        postprocess.run()

        # Clean up temporary directories and/or files created during
        # Schema invocation
        if self.schema.tmpdir is not None:
            self.schema.tmpdir.cleanup()


if __name__ == "__main__":  # pragma: nocover
    sbmod = SegmentAndPostProcess()
    sbmod.run()
