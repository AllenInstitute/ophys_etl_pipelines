import argschema
import pathlib
from marshmallow import post_load
from marshmallow.validate import OneOf


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
            description=("Size of spatial filter kernel; "
                         "if None or zero, no spatial filter applied"))

    kernel_type = argschema.fields.String(
            required=False,
            allow_none=False,
            default='median',
            validation=OneOf(('median', 'mean')),
            description=("Type of spatial smoothing kernel to be "
                         "applied to the video after temporal "
                         "downsampling. (Note: the mean filter will "
                         "downsample each video frame by a factor of "
                         "kernel_size; the median filter does not "
                         "change the size of the video frames)"))

    video_dtype = argschema.fields.String(
            required=False,
            allow_none=False,
            default='uint8',
            validation=OneOf(('uint8', 'uint16')),
            description=("Type to which the output video is cast"))

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
            default=0.0,
            allow_none=False,
            description=("Lower quantile to use when clipping and "
                         "normalizing the video"))

    upper_quantile = argschema.fields.Float(
            required=False,
            default=1.0,
            allow_none=False,
            description=("Upper quantile to use when clipping and "
                         "normalizing the video"))

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
            description=("Factor by which to speed up the movie "
                         "*after downsampling* when writing "
                         "to video (in case you want a "
                         "file that plays back faster)"))

    @post_load
    def check_quantiles(self, data, **kwargs):
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
        allowed = ('.mp4', '.avi', '.tiff', '.tif')
        if output_suffix not in allowed:
            msg = "output_path must have one of these extensions:\n"
            msg += f"{allowed}\n"
            msg += f"you gave {data['output_path']}"
            raise ValueError(msg)
        return data
