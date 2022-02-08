from ophys_etl.modules.video.utils import (
    apply_downsampled_mean_filter_to_video)
from ophys_etl.modules.median_filtered_max_projection.utils import (
    apply_median_filter_to_video)
from functools import partial
import numpy as np


class VideoModuleMixin(object):

    def _get_supplemental_args(self) -> dict:
        """
        Parse self.args and return a dict like
        {'spatial_filter': optional callable spatial filter to apply to video
         'video_dtype': dtype of video
        """

        use_kernel = False
        if self.args['kernel_size'] is not None:
            if self.args['kernel_size'] > 0:
                use_kernel = True

        if use_kernel:
            if self.args['kernel_type'] == 'median':
                spatial_filter = partial(apply_median_filter_to_video,
                                         kernel_size=self.args['kernel_size'])
            else:
                spatial_filter = partial(
                                     apply_downsampled_mean_filter_to_video,
                                     kernel_size=self.args['kernel_size'])
        else:
            spatial_filter = None

        video_dtype = np.dtype(self.args['video_dtype'])

        return {'spatial_filter': spatial_filter,
                'video_dtype': video_dtype}
