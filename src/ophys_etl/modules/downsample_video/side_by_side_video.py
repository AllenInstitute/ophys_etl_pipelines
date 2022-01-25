from ophys_etl.modules.downsample_video.utils import (
    create_side_by_side_video)

from ophys_etl.modules.downsample_video.schemas import (
    DownsampleBaseSchema)

import argschema
import pathlib


class SideBySideDownsamplerSchema(DownsampleBaseSchema):

    left_video_path = argschema.fields.InputFile(
           required=True,
           default=None,
           allow_none=False,
           description=("Path to the input video to be displayed "
                        "in the left panel of the output video"))

    right_video_path = argschema.fields.InputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Path to the input video to be displayed "
                         "in the right panel of the output video"))


class SideBySideDownsampler(argschema.ArgSchemaParser):

    default_schema = SideBySideDownsamplerSchema

    def run(self):
        if self.args['upper_quantile'] is not None:
            quantiles = (self.args['lower_quantile'],
                         self.args['upper_quantile'])
        else:
            quantiles = None

        create_side_by_side_video(
            pathlib.Path(self.args['left_video_path']),
            pathlib.Path(self.args['right_video_path']),
            self.args['input_frame_rate_hz'],
            pathlib.Path(self.args['output_path']),
            self.args['output_frame_rate_hz'],
            self.args['kernel_size'],
            self.args['n_parallel_workers'],
            quality=self.args['quality'],
            quantiles=quantiles,
            reticle=self.args['reticle'],
            speed_up_factor=self.args['speed_up_factor'],
            tmp_dir=self.args['tmp_dir'])


if __name__ == "__main__":
    runner = SideBySideDownsampler()
    runner.run()
