import numpy as np
from argschema import ArgSchema, fields

from ophys_etl.schemas import ROIMasksSchema
from ophys_etl.schemas.fields import H5InputFile


class DemixJobOutputSchema(ArgSchema):
    negative_transient_roi_ids = fields.List(
        fields.Float,
        required=True,
        description=(
            "Path to output h5 file containing the demix traces for "
            "each ROI."
        ),
    )
    negative_baseline_roi_ids = fields.List(
        fields.Float,
        required=True,
        description=(
            "Path to output h5 file containing the demix traces for "
            "each ROI."
        ),
    )


class DemixJobSchema(ArgSchema):
    movie_h5 = H5InputFile(
        required=True,
        description=("Input h5 file containing the motion corrected movie "),
    )
    traces_h5 = H5InputFile(
        required=True,
        description=(
            "Input h5 file containing fluorescence traces and the "
            "associated ROI IDs (in datasets specified by the keys "
            "'input_dataset' and 'roi_field', respectively."
        ),
    )
    output_file = fields.OutputFile(
        required=True,
        description="h5 file to write the results containing demixed traces.",
    )
    roi_masks = fields.List(
        fields.Nested(ROIMasksSchema),
        many=True,
        description=(
            "specifications of individual regions of interest and "
            "their masks"
        ),
    )
