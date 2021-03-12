from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (LogLevel, String, Nested, Boolean,
                              Float, List, Integer,
                              OutputDir, InputFile)

from ophys_etl.schemas.fields import H5InputFile


class MotionBorder(DefaultSchema):
    # TODO: be really certain about how these relate to
    # physical space and then write it here
    x0 = Float(default=0.0, description='')
    x1 = Float(default=0.0, description='')
    y0 = Float(default=0.0, description='')
    y1 = Float(default=0.0, description='')


class Roi(DefaultSchema):
    mask = List(
        List(Boolean),
        required=True,
        description="raster mask")
    y = Integer(
        required=True,
        description="y position (pixels) of mask's bounding box")
    x = Integer(
        required=True,
        description="x position (pixels) of mask's bounding box")
    width = Integer(
        required=True,
        description="width (pixels)of mask's bounding box")
    height = Integer(
        required=True,
        description="height (pixels) of mask's bounding box")
    valid = Boolean(
        default=True,
        description="Is this Roi known to be valid?")
    id = Integer(
        required=True,
        description="unique integer identifier for this Roi")
    mask_page = Integer(
        default=-1,
        description="")


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
        Roi,
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
                     "traces"))  # TODO rename these to _path
    roi_trace_file = H5FileExists(
        required=True,
        description="path to output h5 file containing roi traces")
    exclusion_labels = Nested(
        ExclusionLabel,
        many=True,
        description=("a report of roi-wise problems detected "
                     "during extraction"))
