from ophys_etl.modules.median_filtered_max_projection.utils import (
    apply_median_filter_to_video)

from ophys_etl.modules.video.utils import (
    create_downsampled_video,
    apply_mean_filter_to_video)

from ophys_etl.modules.video.schemas import (
    VideoBaseSchema)

from functools import partial
import numpy as np
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

        use_kernel = False
        if self.args['kernel_size'] is not None:
            if self.args['kernel_size'] > 0:
                use_kernel = True

        if use_kernel:
            if self.args['kernel_type'] == 'median':
                spatial_filter = partial(apply_median_filter_to_video,
                                         kernel_size=self.args['kernel_size'])
            else:
                spatial_filter = partial(apply_mean_filter_to_video,
                                         kernel_size=self.args['kernel_size'])
        else:
            spatial_filter = None

        if self.args['video_dtype'] == 'uint8':
            video_dtype = np.uint8
        else:
            video_dtype = np.uint16

        create_downsampled_video(
            pathlib.Path(self.args['video_path']),
            self.args['input_frame_rate_hz'],
            pathlib.Path(self.args['output_path']),
            self.args['output_frame_rate_hz'],
            spatial_filter,
            self.args['n_parallel_workers'],
            quality=self.args['quality'],
            quantiles=quantiles,
            reticle=self.args['reticle'],
            speed_up_factor=self.args['speed_up_factor'],
            tmp_dir=self.args['tmp_dir'],
            video_dtype=video_dtype)


if __name__ == "__main__":
    downsampler = VideoGenerator()
    downsampler.run()
