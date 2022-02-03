from ophys_etl.modules.median_filtered_max_projection.utils import (
    apply_median_filter_to_video)

from ophys_etl.modules.video.utils import (
    create_side_by_side_video)

from ophys_etl.modules.video.schemas import (
    VideoBaseSchema)

from functools import partial
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


class SideBySideVideoGenerator(argschema.ArgSchemaParser):

    default_schema = SideBySideVideoSchema

    def run(self):
        if self.args['upper_quantile'] is not None:
            quantiles = (self.args['lower_quantile'],
                         self.args['upper_quantile'])
        else:
            quantiles = None

        if self.args['kernel_size'] is not None:
            spatial_filter = partial(apply_median_filter_to_video,
                                     kernel_size=self.args['kernel_size'])
        else:
            spatial_filter = None

        create_side_by_side_video(
            pathlib.Path(self.args['left_video_path']),
            pathlib.Path(self.args['right_video_path']),
            self.args['input_frame_rate_hz'],
            pathlib.Path(self.args['output_path']),
            self.args['output_frame_rate_hz'],
            spatial_filter,
            self.args['n_parallel_workers'],
            quality=self.args['quality'],
            quantiles=quantiles,
            reticle=self.args['reticle'],
            speed_up_factor=self.args['speed_up_factor'],
            tmp_dir=self.args['tmp_dir'])


if __name__ == "__main__":
    runner = SideBySideVideoGenerator()
    runner.run()