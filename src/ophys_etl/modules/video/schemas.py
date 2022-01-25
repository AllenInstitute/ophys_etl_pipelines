import argschema
import pathlib
from marshmallow import post_load


class VideoBaseSchema(argschema.ArgSchema):

    output_path = argschema.fields.OutputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Path to video file that will be written; "
                         "must be .avi or .mp4 file"))

    kernel_size = argschema.fields.Int(
            required=False,
            allow_none=True,
            default=3,
            description=("Radius of median filter kernel; "
                         "if None, no median filter applied"))

    input_frame_rate_hz = argschema.fields.Float(
            required=True,
            default=None,
            allow_none=False,
            description=("Frame rate of input video in Hz"))

    output_frame_rate_hz = argschema.fields.Float(
            required=True,
            default=None,
            allow_none=False,
            description=("Frame rate of output video in Hz "
                         "used for downsampling"))

    reticle = argschema.fields.Boolean(
            required=False,
            default=True,
            description=("If True, a grid of red lines is added "
                         "to the output video to guide the eye"))

    n_parallel_workers = argschema.fields.Int(
            required=False,
            default=16,
            description=("Number of parallel processes to spawn "
                         "when applying expensive spatial filtering "
                         "to the movie"))

    lower_quantile = argschema.fields.Float(
            required=False,
            default=None,
            allow_none=True,
            description=("Lower quantile to use when clipping and "
                         "normalizing the video; if quantiles are None, "
                         "will use the min and max values of the video"))

    upper_quantile = argschema.fields.Float(
            required=False,
            default=None,
            allow_none=True,
            description=("Upper quantile to use when clipping and "
                         "normalizing the video; if quantiles are None, "
                         "will use the min and max values of the video"))

    tmp_dir = argschema.fields.OutputDir(
            required=False,
            default=None,
            allow_none=True,
            description=("Path to scratch dir in which to write intermediate "
                         "output files"))

    quality = argschema.fields.Int(
            required=False,
            default=5,
            allow_none=False,
            description=("Quality parameter to be passed to ffmpeg; must be "
                         "0-9 inclusive"))

    speed_up_factor = argschema.fields.Int(
            required=False,
            default=8,
            allow_none=False,
            description=("Factor by which to speed up output movie"))

    @post_load
    def check_quantiles(self, data, **kwargs):
        valid = True
        if data['upper_quantile'] is None:
            if not data['lower_quantile'] is None:
                valid = False
        if data['lower_quantile'] is None:
            if not data['upper_quantile'] is None:
                valid = False
        if not valid:
            msg = 'If upper_quantile is None, lower_quantile must be None '
            msg += 'and vice-versa'
            raise ValueError(msg)
        if data['upper_quantile'] is not None:
            if data['upper_quantile'] <= data['lower_quantile']:
                msg = 'upper_quantile must be > lower_quantile'
                raise ValueError(msg)
        return data

    @post_load
    def check_quality(self, data, **kwargs):
        if data['quality'] > 9:
            raise ValueError('quality must be <= 9')
        return data

    @post_load
    def check_output_path(self, data, **kwargs):
        output_path = pathlib.Path(data['output_path'])
        output_suffix = output_path.suffix
        if output_suffix not in ('.mp4', '.avi'):
            msg = "output_path must be an .mp4 or .avi file\n"
            msg += f"you gave {data['output_path']}"
            raise ValueError(msg)
        return data
