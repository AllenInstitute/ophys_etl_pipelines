import h5py
import argschema
from marshmallow import post_load, ValidationError

from ophys_etl.schemas.fields import InputOutputFile
from ophys_etl.modules.segmentation.processing_log import \
    SegmentationProcessingLog


class FilterBaseSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")

    log_path = InputOutputFile(
            required=True,
            description="Path to HDF5 log for input/output ")

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

    rois_group = argschema.fields.String(
        required=False,
        default=None,
        allow_none=True,
        description=("name of hdf5 group from which to take the ROIs to "
                     "merge. If not provided, will take the last group."))

    @post_load
    def check_for_rois(self, data, **kwargs):
        qcfile = SegmentationProcessingLog(data["log_path"])
        if data["rois_group"] is None:
            data["rois_group"] = qcfile.get_last_group()
        with h5py.File(qcfile.path, "r") as f:
            if "rois" not in f[data["rois_group"]]:
                raise ValidationError(f"group {data['rois_group']} does not "
                                      "have dataset 'rois' in file "
                                      f"{data['log_path']}")
        return data


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
