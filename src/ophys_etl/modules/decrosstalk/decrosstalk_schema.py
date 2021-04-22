import marshmallow.exceptions
import argschema

from ophys_etl.schemas import ExtractROISchema


class PlaneSchema(argschema.schemas.DefaultSchema):

    ophys_experiment_id = argschema.fields.Int(
                             description='experiment ID',
                             required=True)

    output_roi_trace_file = argschema.fields.OutputFile(
                       description='Path to ROI trace file',
                       required=True)

    output_neuropil_trace_file = argschema.fields.OutputFile(
                             description='Path to neuropil trace file',
                             required=True)

    motion_corrected_stack = argschema.fields.InputFile(
                        description='path to motion corrected movie',
                        required=True)

    motion_border = argschema.fields.Dict(description='motion border',
                                          required=True)

    rois = argschema.fields.Nested(
                ExtractROISchema,
                description='regions of interest',
                required=True,
                many=True)

    maximum_projection_image_file = argschema.fields.InputFile(
                        description='path to maximum projection image',
                        required=True)


class PlanePairSchema(argschema.schemas.DefaultSchema):

    ophys_imaging_plane_group_id = argschema.fields.Int(
                                    description='group ID',
                                    required=True)

    group_order = argschema.fields.Int(
                      description='order of group in session',
                      required=True)

    planes = argschema.fields.List(
                 argschema.fields.Nested(PlaneSchema),
                 description='planes',
                 required=True,
                 cli_as_single_argument=True)


class DecrosstalkInputSchema(argschema.ArgSchema):

    ophys_session_id = argschema.fields.Int(
                           description='ophys_session_id',
                           required=True)

    qc_output_dir = argschema.fields.OutputDir(
                        description='path to intermediate output dir',
                        required=True)

    coupled_planes = argschema.fields.List(
                        argschema.fields.Nested(PlanePairSchema),
                        description='list of plane pairs',
                        required=True,
                        cli_as_single_argument=True)

    log_level = argschema.fields.LogLevel(default='INFO')


def validate_list_of_ints(input_list):
    """
    Validator for DecrosstalkOutputSchema.
    Checks that input_list is, indeed, a list of ints.
    """
    if not isinstance(input_list, list):
        return False
    for ii in input_list:
        if not isinstance(ii, int):
            msg = 'Did not contain ints'
            raise marshmallow.exceptions.ValidationError(msg)
    return True


class PlaneOutputSchema(argschema.schemas.DefaultSchema):

    ophys_experiment_id = argschema.fields.Int(
                             description='experiment ID',
                             required=True)

    output_roi_trace_file = argschema.fields.OutputFile(
                               description='Path to ROI trace file',
                               required=True)

    output_neuropil_trace_file = argschema.fields.OutputFile(
                                   description='Path to neuropil trace file',
                                   required=True)

    decrosstalk_invalid_raw = argschema.fields.List(
                                  argschema.fields.Field,
                                  description='IDs of ROIs ruled invalid '
                                              'based on raw trace',
                                  required=True,
                                  validate=validate_list_of_ints)

    decrosstalk_invalid_raw_active = argschema.fields.List(
                                  argschema.fields.Field,
                                  description='IDs of ROIs ruled invalid '
                                              'based on raw active trace',
                                  required=True,
                                  validate=validate_list_of_ints)

    decrosstalk_invalid_unmixed = argschema.fields.List(
                                        argschema.fields.Field,
                                        description='IDs of ROIs ruled '
                                                    'invalid based on '
                                                    'unmixed trace',
                                        required=True,
                                        validate=validate_list_of_ints)

    decrosstalk_invalid_unmixed_active = argschema.fields.List(
                                           argschema.fields.Field,
                                           description='IDs of ROIs ruled '
                                                       'invalid based on'
                                                       'unmixed active trace',
                                           required=True,
                                           validate=validate_list_of_ints)

    decrosstalk_ghost = argschema.fields.List(
                                      argschema.fields.Field,
                                      description='IDs of ROIs ruled invalid '
                                                  'to be ghosts',
                                      required=True,
                                      validate=validate_list_of_ints)


class PlanePairOutputSchema(argschema.schemas.DefaultSchema):

    ophys_imaging_plane_group_id = argschema.fields.Int(
                                    description='group ID',
                                    required=True)

    group_order = argschema.fields.Int(
                      description='order of group in session',
                      required=True)

    planes = argschema.fields.List(
                 argschema.fields.Nested(PlaneOutputSchema),
                 description='planes',
                 required=True,
                 cli_as_single_argument=True)


class DecrosstalkOutputSchema(argschema.ArgSchema):
    # All Lists are lists of argschema.fields.Field because
    # this was the only way I could get argschema to not
    # automatically cast non-int numeric types into ints.
    # The call to validate_list_of_ints will make sure that
    # the lists do, indeed, need to contain ints.

    ophys_session_id = argschema.fields.Int(
                           description='ophys_session_id',
                           required=True)

    coupled_planes = argschema.fields.List(
                            argschema.fields.Nested(PlanePairOutputSchema),
                            description='list of plane pairs',
                            required=True,
                            cli_as_single_argument=True)
