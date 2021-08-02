import h5py
import argschema
from marshmallow import post_load, ValidationError
from marshmallow.validate import OneOf

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
        qcfile = SegmentationProcessingLog(data["log_path"], read_only=True)
        if data["rois_group"] is None:
            data["rois_group"] = qcfile.get_last_group()
        with h5py.File(qcfile.path, "r") as f:
            if "rois" not in f[data["rois_group"]]:
                raise ValidationError(f"group {data['rois_group']} does not "
                                      "have dataset 'rois' in file "
                                      f"{data['log_path']}")
        return data


class MetricBaseSchema(FilterBaseSchema):

    graph_input = argschema.fields.InputFile(
            default=None,
            required=True,
            allow_none=False,
            description=('path to pkl file containing the graph '
                         'that will be used to generate the metric image'))

    attribute_name = argschema.fields.String(
            default='filtered_hnc_Gaussian',
            required=True,
            allow_none=False,
            description=('attribute of graph that will be used to '
                         'generate the metric image'))

    @post_load
    def check_graph_name(self, data, **kwargs):
        if not str(data['graph_input']).endswith('pkl'):
            msg = f"\n{data['graph_input']} does not appear "
            msg += "to be a pkl file\n"
            raise RuntimeError(msg)
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


class StatFilterSchema(MetricBaseSchema):

    stat_name = argschema.fields.String(
            default=None,
            required=True,
            allow_none=False,
            validation=OneOf(['mean', 'median']),
            description=('metric statistic on which to filter'))

    min_value = argschema.fields.Float(
            default=None,
            required=False,
            allow_none=True,
            description=("minimum value of stat allowed for a valid ROI"))

    max_value = argschema.fields.Float(
            default=None,
            required=False,
            allow_none=True,
            description=("maximum value of stat allowed for a valid ROI"))

    @post_load
    def check_stat_filter_fields(self, data, **kwargs):
        msg = ''
        is_valid = True
        if data['min_value'] is None and data['max_value'] is None:
            msg += "\nmin_value and max_value are both None; "
            msg += "must specify at least one\n"
            is_valid = False

        if data['min_value'] is not None and data['max_value'] is not None:
            if data['min_value'] > data['max_value']:
                msg += "\nmin_value > max_value\n"
                is_valid = False

        if not is_valid:
            raise RuntimeError(msg)

        return data


class ZvsBackgroundSchema(MetricBaseSchema):

    min_z = argschema.fields.Float(
            default=None,
            required=True,
            allow_none=False,
            description=("minimum z-value above background of valid ROI"))

    n_background_factor = argschema.fields.Int(
            default=2,
            required=False,
            allow_none=False,
            description=("select N_BACKGROUND_FACTOR*ROI.AREA pixels "
                         "when constructing population of background "
                         "pixels to use in calculating z-score"))

    n_background_minimum = argschema.fields.Int(
            default=100,
            required=False,
            allow_none=False,
            description=("minimum number of background pixels to use "
                         "when selecting population of background "
                         "pixels to use in calculating z-score "
                         "(in case ROI.AREA is small)"))
