import argschema
from pathlib import Path

from ophys_etl.schemas import ExtractROISchema


class BridgeInputSchema(argschema.ArgSchema):
    input_file = argschema.fields.InputFile(
        required=True,
        description=("an output from BinarizerAndROICreator in "
                     "convert_rois.py"))
    storage_directory = argschema.fields.OutputDir(
        required=True,
        description="the intended destination directory for the traces")
    # NOTE these schema field names match convert_rois and not AllenSDK
    motion_corrected_video = argschema.fields.Str(
        required=True,
        validate=lambda x: Path(x).exists(),
        description=("Path to motion corrected video file *.h5"))
    motion_correction_values = argschema.fields.InputFile(
        required=True,
        description=("Path to motion correction values for each frame "
                     "stored in .csv format. This .csv file is expected to"
                     "have a header row of either:\n"
                     "['framenumber','x','y','correlation','kalman_x',"
                     "'kalman_y']\n['framenumber','x','y','correlation',"
                     "'input_x','input_y','kalman_x',"
                     "'kalman_y','algorithm','type']"))


class MotionBorderSchema(argschema.schemas.DefaultSchema):
    x0 = argschema.fields.Float()
    x1 = argschema.fields.Float()
    y0 = argschema.fields.Float()
    y1 = argschema.fields.Float()


class BridgeOutputSchema(argschema.schemas.DefaultSchema):
    # NOTE these schema field names match AllenSDK and not convert_rois
    motion_border = argschema.fields.Nested(
            MotionBorderSchema,
            required=True)
    storage_directory = argschema.fields.OutputDir(
            required=True)
    motion_corrected_stack = argschema.fields.Str(
            validate=lambda x: Path(x).exists(),
            required=True)
    rois = argschema.fields.Nested(
            ExtractROISchema,
            required=True,
            many=True)
    log_0 = argschema.fields.InputFile(
            required=True)
