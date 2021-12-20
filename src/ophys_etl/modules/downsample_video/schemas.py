import argschema
from marshmallow import post_load


class DownsampleBaseSchema(argschema.ArgSchema):

    output_path = argschema.fields.OutputFile(
            required=True,
            default=None,
            allow_none=False)

    kernel_size = argschema.fields.Int(
            required=False,
            allow_none=True,
            default=3,
            description=("Radius of median filter kernel; "
                         "if None, no median filter applied"))

    input_frame_rate_hz = argschema.fields.Float(
            required=True,
            default=None,
            allow_none=False)


    output_frame_rate_hz = argschema.fields.Float(
            required=True,
            default=None,
            allow_none=False)

    reticle = argschema.fields.Boolean(
            required=False,
            default=True)

    n_parallel_workers = argschema.fields.Int(
            required=False,
            default=16)

    lower_quantile = argschema.fields.Float(
            required=False,
            default=None,
            allow_none=True)

    upper_quantile = argschema.fields.Float(
            required=False,
            default=None,
            allow_none=True)

    tmp_dir = argschema.fields.OutputDir(
            required=False,
            default=None,
            allow_none=True)

    quality = argschema.fields.Int(
            required=False,
            default=5,
            allow_none=False)

    speed_up_factor = argschema.fields.Int(
            required=False,
            default=8,
            allow_none=False)


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
