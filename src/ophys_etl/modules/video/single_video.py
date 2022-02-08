from ophys_etl.modules.video.cli_mixins import VideoModuleMixin

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


class VideoGenerator(argschema.ArgSchemaParser,
                     VideoModuleMixin):

    default_schema = VideoSchema

    def run(self):
        supplemental_args = self._get_supplemental_args()

        create_downsampled_video(
            input_path=pathlib.Path(self.args['video_path']),
            input_hz=self.args['input_frame_rate_hz'],
            output_path=pathlib.Path(self.args['output_path']),
            output_hz=self.args['output_frame_rate_hz'],
            spatial_filter=supplemental_args['spatial_filter'],
            n_processors=self.args['n_parallel_workers'],
            quality=self.args['quality'],
            quantiles=(self.args['lower_quantile'],
                       self.args['upper_quantile']),
            reticle=self.args['reticle'],
            speed_up_factor=self.args['speed_up_factor'],
            tmp_dir=self.args['tmp_dir'],
            video_dtype=supplemental_args['video_dtype'])


if __name__ == "__main__":
    downsampler = VideoGenerator()
    downsampler.run()
