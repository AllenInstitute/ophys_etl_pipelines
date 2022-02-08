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
        {'quantiles': optional tuple of quantiles to use in clipping video
         'spatial_filter': optional callable spatial filter to apply to video
         'video_dtype': dtype of video
        """
        if self.args['upper_quantile'] is not None:
            quantiles = (self.args['lower_quantile'],
                         self.args['upper_quantile'])
        else:
            quantiles = (0.0, 1.0)

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

        if self.args['video_dtype'] == 'uint8':
            video_dtype = np.uint8
        else:
            video_dtype = np.uint16

        return {'quantiles': quantiles,
                'spatial_filter': spatial_filter,
                'video_dtype': video_dtype}
