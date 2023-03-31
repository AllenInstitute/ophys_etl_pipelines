from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (LogLevel, String, Nested, Float,
                              OutputDir, InputFile, OutputFile)

from ophys_etl.schemas.fields import H5InputFile
from ophys_etl.schemas import ExtractROISchema


class MotionBorder(DefaultSchema):
    x0 = Float(default=0.0, description='npixel border from left')
    x1 = Float(default=0.0, description='npixel border from right')
    y0 = Float(default=0.0, description='npixel border from top')
    y1 = Float(default=0.0, description='npixel border from bottom')


class ExclusionLabel(DefaultSchema):
    roi_id = String(required=True)
    exclusion_label_name = String(required=True)


class TraceExtractionInputSchema(ArgSchema):
    log_level = LogLevel(
        default='INFO',
        description="set the logging level of the module")
    motion_border = Nested(
        MotionBorder,
        required=True,
        description=("border widths - pixels outside the border are "
                     "considered invalid"))
    storage_directory = OutputDir(
        required=True,
        description="used to set output directory")
    motion_corrected_stack = H5InputFile(
        required=True,
        description="path to h5 file containing motion corrected image stack")
    rois = Nested(
        ExtractROISchema,
        many=True,
        description="specifications of individual regions of interest")
    log_0 = InputFile(
        required=False,
        description=("path to motion correction output csv. "
                     "NOTE: not used, but provided by LIMS schema."))


class H5FileExists(H5InputFile):
    pass


class TraceExtractionOutputSchema(DefaultSchema):
    neuropil_trace_file = H5FileExists(
        required=True,
        description=("path to output h5 file containing neuropil "
                     "traces"))
    roi_trace_file = H5FileExists(
        required=True,
        description="path to output h5 file containing roi traces")
    neuropil_mask_file = OutputFile(
        required=True,
        description="path to output h5 file containing neuropil masks")
    exclusion_labels = Nested(
        ExclusionLabel,
        many=True,
        description=("a report of roi-wise problems detected "
                     "during extraction"))
