from argschema.schemas import ArgSchema, DefaultSchema
from argschema.fields import (Int, Float, Str, Nested, InputFile, OutputDir,
                              Dict, List, InputDir)


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
    log_level = Str(required=False, default="INFO")
    platform_json_path = InputFile(
        required=False,
        default=None,
        allow_none=True,
        description=("Full path to the platform.json file. "
                     "This file actually contains all of the "
                     "parameters needed to perform the TIFF "
                     "splitting operation. In the future, we "
                     "should be able to replace the other "
                     "input parameters with code that reads "
                     "them directly from this file"))
    data_upload_dir = InputDir(
        required=False,
        default=None,
        allow_none=True,
        description=("Directory to which the raw data files "
                     "are uploaded by the platform team for "
                     "processing. The code will look in here "
                     "for the data products specified in the "
                     "platform.json file."))
    session_id = Int(
        required=False,
        default=None,
        allow_none=True,
        description=("ophys_session_id; used for naming output "
                     "files whose names are not explicitly listed "
                     "in the input.json"))
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
    dump_every = Int(
        required=False,
        default=3000,
        description=("Write timeseries data to scratch files every "
                     "dump_every frames"))
    tmp_dir = OutputDir(
        required=False,
        default='/tmp',
        description=("Directory where the temporary files created during "
                     "timeseries splitting are written. If None, each "
                     "OphysExperiment will create its own temporary "
                     "directory in the output directory for the final "
                     "HDF5 file"))


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
