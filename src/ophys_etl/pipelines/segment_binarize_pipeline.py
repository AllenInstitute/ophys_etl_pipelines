import argschema
from ophys_etl.transforms.suite2p_wrapper import Suite2PWrapper
from ophys_etl.transforms.convert_rois import BinarizerAndROICreator
import tempfile
import pathlib
import json


class SegmentBinarizeSchema(argschema.ArgSchema):
    motion_corrected_video = argschema.fields.InputFile(
        required=True,
        description="hdf5 motion-correced video")
    motion_correction_values = argschema.fields.InputFile(
        required=True,
        description="motion correction values in a csv")


class SegmentBinarize(argschema.ArgSchemaParser):
    default_schema = SegmentBinarizeSchema

    def run(self):
        tmpdir = tempfile.TemporaryDirectory()
        Suite2P_output = pathlib.Path(tmpdir.name) / "Suite2P_output.json"

        # segment with Suite2P, almost all default args
        segment_args = {
                "h5py": self.args['motion_corrected_video'],
                "output_dir": tmpdir.name,
                "output_json": str(Suite2P_output),
                "retain_files": ["stat.npy"],
                "bin_size": 115,
                "log_level": self.args['log_level'],
                }
        segment = Suite2PWrapper(input_data=segment_args, args=[])
        segment.run()

        # binarize and output LIMS format
        with open(Suite2P_output, "r") as f:
            stat = json.load(f)['output_files']['stat.npy']

        binarize_args = {"suite2p_stat_path": stat}

        for k in ['output_json', 'motion_corrected_video',
                  'motion_correction_values', 'log_level']:
            binarize_args[k] = self.args[k]

        binarize = BinarizerAndROICreator(input_data=binarize_args, args=[])
        binarize.binarize_and_create()
        tmpdir.cleanup()


if __name__ == "__main__":  # pragma: nocover
    sbmod = SegmentBinarize()
    sbmod.run()
