import json
import pathlib
import tempfile

import marshmallow
import argschema


from ophys_etl.transforms.convert_rois import (
    BinarizeAndCreateROIsInputSchema, BinarizerAndROICreator)
from ophys_etl.transforms.suite2p_wrapper import (Suite2PWrapper,
                                                  Suite2PWrapperSchema)


class SegmentBinarizeSchema(argschema.ArgSchema):
    suite2p_args = argschema.fields.Nested(Suite2PWrapperSchema,
                                           required=True)
    convert_args = argschema.fields.Nested(BinarizeAndCreateROIsInputSchema,
                                           required=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tmpdir = None
        self.tmp_tempfile = None

    @marshmallow.pre_load
    def setup_default_suite2p_args(self, data: dict, **kwargs) -> dict:

        if "output_dir" not in data["suite2p_args"]:
            self.tmpdir = tempfile.TemporaryDirectory()
            data["suite2p_args"]["output_dir"] = self.tmpdir.name

        if "output_json" not in data["suite2p_args"]:
            Suite2p_output = (pathlib.Path(data["suite2p_args"]["output_dir"])
                              / "Suite2P_output.json")
            data["suite2p_args"]["output_json"] = str(Suite2p_output)

        if "bin_size" not in data["suite2p_args"]:
            data["suite2p_args"]["bin_size"] = 115

        data["suite2p_args"]["log_level"] = data["log_level"]

        if "convert_args" in data:
            data["convert_args"]["motion_corrected_video"] = (
                data["suite2p_args"]["h5py"])

            if "output_json" not in data["convert_args"]:
                data["convert_args"]["output_json"] = data["output_json"]

            # Ugly hack to provide a temporary file for the suite2p_stat_path
            # Because argschema schemas apparently don't handle "exclude" or
            # "only" marshmallow params properly
            self.tmp_tempfile = tempfile.NamedTemporaryFile()
            data["convert_args"]["suite2p_stat_path"] = self.tmp_tempfile.name

        return data

    @marshmallow.post_load
    def setup_default_convert_args(self, data: dict, **kwargs) -> dict:
        """Setup convert args (relevant if no convert_args were passed)"""

        if "convert_args" not in data:
            data["convert_args"] = dict()
            data["convert_args"]["motion_corrected_video"] = (
                data["suite2p_args"]["h5py"])
            data["convert_args"]["output_json"] = data["output_json"]
            data["convert_args"]["log_level"] = data["log_level"]

        return data


class SegmentBinarize(argschema.ArgSchemaParser):
    default_schema = SegmentBinarizeSchema

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

        binarize = BinarizerAndROICreator(input_data=convert_args, args=[])
        binarize.binarize_and_create()

        # Clean up temporary directories and/or files created during
        # Schema invocation
        if hasattr(self.schema, 'tmpdir') and self.schema.tmpdir:
            self.schema.tmpdir.cleanup()
        if hasattr(self.schema, 'tmp_tmpfile') and self.schema.tmp_tmpfile:
            self.schema.tmp_tmpfile.close()


if __name__ == "__main__":  # pragma: nocover
    sbmod = SegmentBinarize()
    sbmod.run()
