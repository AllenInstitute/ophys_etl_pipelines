from ophys_etl.modules.video.utils import (
    create_downsampled_video)

from ophys_etl.modules.video.schemas import (
    VideoBaseSchema)

import argschema
import pathlib


class VideoSchema(VideoBaseSchema):

    video_path = argschema.fields.InputFile(
           required=True,
           default=None,
           allow_none=False,
           description="Path to the input video file")


class VideoGenerator(argschema.ArgSchemaParser):

    default_schema = VideoSchema

    def run(self):
        if self.args['upper_quantile'] is not None:
            quantiles = (self.args['lower_quantile'],
                         self.args['upper_quantile'])
        else:
            quantiles = None

        create_downsampled_video(
            pathlib.Path(self.args['video_path']),
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
    downsampler = VideoGenerator()
    downsampler.run()
