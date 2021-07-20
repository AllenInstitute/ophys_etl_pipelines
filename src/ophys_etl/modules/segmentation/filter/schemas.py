import argschema
from marshmallow import post_load


class FilterBaseSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")

    roi_input = argschema.fields.InputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Path to the JSON file containing the ROIs "
                         "to be filtered"))

    roi_output = argschema.fields.OutputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Path to the JSON file where the filtered ROIs "
                         "will be written"))

    roi_log_path = argschema.fields.OutputFile(
            required=True,
            default=None,
            allow_none=False,
            description=("Path to HDF5 file where a record of why "
                         "ROIs were flagged as invalid will be kept"))

    pipeline_stage = argschema.fields.String(
            required=True,
            default=None,
            allow_none=False,
            description=("A tag denoting what stage in the segmentation "
                         "pipeline this filter represents. This will be "
                         "appended to the 'reason' an ROI was invalidated "
                         "in the log file. It is meant to differentiate "
                         "between, e.g., filtering that happens between "
                         "growth and merging and filtering that happens "
                         "after merging"))


class AreaFilterSchema(FilterBaseSchema):

    max_area = argschema.fields.Int(
            default=None,
            required=False,
            allow_none=True,
            description=("maximum area an ROI can have and still be valid"))

    min_area = argschema.fields.Int(
            default=None,
            required=False,
            allow_none=True,
            description=("minimum area an ROI can have and still be valid"))

    @post_load
    def check_min_and_max(self, data, **kwargs):
        if data['min_area'] is None and data['max_area'] is None:
            msg = "min_area and max_area are both None; "
            msg += "must specify at least one"
            raise RuntimeError(msg)
        return data
