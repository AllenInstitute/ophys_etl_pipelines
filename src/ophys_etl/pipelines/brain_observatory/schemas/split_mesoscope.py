from argschema.schemas import ArgSchema, DefaultSchema
from argschema.fields import (Int, Float, Str, Nested, InputFile, OutputDir,
                              Dict, List)


class ExperimentPlane(DefaultSchema):
    experiment_id = Int(
        required=True,
        description="Ophys experiment id.")
    storage_directory = OutputDir(
        required=True,
        description="Folder for output files.")
    roi_index = Int(
        required=True,
        description="Index of ROI in the scanimage metadata.")
    scanfield_z = Float(
        required=True,
        description="Z value of the scanfield for this experiment plane.")
    resolution = Float(
        required=True,
        description="Pixel size in microns.")
    offset_x = Int(
        required=True,
        description="X offset of image from gold reticle.")
    offset_y = Int(
        required=True,
        description="Y offset of image from gold reticle.")
    rotation = Float(
        required=True,
        description="Rotation of image relative to gold reticle.")


class PlaneGroup(DefaultSchema):
    local_z_stack_tif = InputFile(
        required=True,
        description="Full path to local z stack tiff file for this group.")
    column_z_stack_tif = InputFile(
        description="Full path to column z stack tiff file.")
    ophys_experiments = Nested(
        ExperimentPlane,
        many=True)


class InputSchema(ArgSchema):
    depths_tif = InputFile(
        required=True,
        description="Full path to depth 2p tiff file.")
    surface_tif = InputFile(
        required=True,
        description="Full path to surface 2p tiff file.")
    timeseries_tif = InputFile(
        required=True,
        description="Full path to timeseries tiff file.")
    storage_directory = OutputDir(
        required=True,
        description="Folder for column stack outputs.")
    plane_groups = Nested(
        PlaneGroup,
        many=True)
    compression_level = Int(
        default=4,
        description="Gzip compression level to use. 1-9.")
    test_mode = Int(
        default=0,
        description=("Flag to run without actually splitting data. For testing"
                     " runner mechanism and metadata. Testing of splitting "
                     "is handled in testing for the mesoscope_2p package."))


class TiffMetadataOutput(DefaultSchema):
    input_tif = Str()
    roi_metadata = Dict()
    scanimage_metadata = Dict()


class ImageFileOutput(DefaultSchema):
    filename = Str()
    resolution = Float(
        required=True,
        description="Pixel size in microns.")
    offset_x = Int(
        required=True,
        description="X offset of image from gold reticle.")
    offset_y = Int(
        required=True,
        description="Y offset of image from gold reticle.")
    rotation = Float(
        required=True,
        description="Rotation of image relative to gold reticle.")
    width = Int()
    height = Int()


class ExperimentOutput(DefaultSchema):
    experiment_id = Int(
        required=True)
    local_z_stack = Nested(ImageFileOutput, required=True)
    surface_2p = Nested(ImageFileOutput, required=True)
    depth_2p = Nested(ImageFileOutput, required=True)
    timeseries = Nested(ImageFileOutput, required=True)
    sync_offset = Int()
    sync_stride = Int()


class OutputSchema(DefaultSchema):
    column_stacks = Nested(
        ImageFileOutput,
        many=True)
    file_metadata = Nested(
        TiffMetadataOutput,
        many=True)
    ready_to_archive = List(
        Str,
        required=True)
    experiment_output = Nested(
        ExperimentOutput,
        many=True)