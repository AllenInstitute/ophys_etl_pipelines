from ophys_etl.modules.video.cli_mixins import VideoModuleMixin

from ophys_etl.modules.video.utils import (
    create_side_by_side_video)

from ophys_etl.modules.video.schemas import (
    VideoBaseSchema)

import argschema
import pathlib


class SideBySideVideoSchema(VideoBaseSchema):

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


class SideBySideVideoGenerator(argschema.ArgSchemaParser,
                               VideoModuleMixin):

    default_schema = SideBySideVideoSchema

    def run(self):

        supplemental_args = self._get_supplemental_args()

        create_side_by_side_video(
            left_video_path=pathlib.Path(self.args['left_video_path']),
            right_video_path=pathlib.Path(self.args['right_video_path']),
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
    runner = SideBySideVideoGenerator()
    runner.run()
