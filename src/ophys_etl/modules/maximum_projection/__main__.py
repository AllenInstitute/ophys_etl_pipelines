import h5py
import PIL.Image
import argschema
from marshmallow import post_load

from ophys_etl.modules.maximum_projection.utils import (
    generate_max_projection,
    scale_to_uint8)


class MaximumProjectionSchema(argschema.ArgSchema):

    video_path = argschema.fields.InputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Path to HDF5 containing video data"))

    image_path = argschema.fields.OutputFile(
            required=True,
            default=None,
            allow_nonw=False,
            description=("Path to png file where image will be stored"))

    input_frame_rate = argschema.fields.Float(
            required=True,
            default=None,
            allow_none=False,
            description=("frame rate (in Hz) of video"))

    downsampled_frame_rate = argschema.fields.Float(
            required=False,
            default=4.0,
            allow_none=False,
            description=("frame rate (in Hz) to which video is "
                         "down sampled before applying median filter"))

    median_filter_kernel_size = argschema.fields.Integer(
            required=False,
            default=3,
            allow_none=False,
            description=("Side length of square kernel used in median filter"))

    n_parallel_workers = argschema.fields.Integer(
            required=False,
            default=32,
            allow_none=False,
            description=("Number of processes to use when divvying up work"))

    @post_load
    def check_png_path(self, data, **kwargs):
        if not data['image_path'].endswith('png'):
            msg = f"You gave image_path={data['image_path']}\n"
            msg += "must be a path to a .png file"
            raise ValueError(msg)
        return data


class MaximumProjectionRunner(argschema.ArgSchemaParser):
    default_schema = MaximumProjectionSchema

    def run(self):

        img = generate_max_projection(
                    self.args['video_path'],
                    self.args['input_frame_rate'],
                    self.args['downsampled_frame_rate'],
                    self.args['median_filter_kernel_size'],
                    self.args['n_parallel_workers'])

        img = PIL.Image.fromarray(scale_to_uint8(img))
        img.save(self.args['image_path'])


if __name__ == "__main__":
    runner = MaximumProjectionRunner()
    runner.run()
