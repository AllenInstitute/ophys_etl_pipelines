from argschema import ArgSchema, fields

from ophys_etl.schemas import ROIMasksSchema
from ophys_etl.schemas.fields import H5InputFile

EXCLUDE_LABELS = [
    "union",
    "duplicate",
    "motion_border",
    "decrosstalk_ghost",
    "decrosstalk_invalid_raw",
    "decrosstalk_invalid_raw_active",
    "decrosstalk_invalid_unmixed",
    "decrosstalk_invalid_unmixed_active",
]


class DemixJobOutputSchema(ArgSchema):
    negative_transient_roi_ids = fields.List(
        fields.Float,
        required=True,
        description=("ROI IDs detected to contain negative transients"),
    )
    negative_baseline_roi_ids = fields.List(
        fields.Float,
        required=True,
        description=("ROI IDs detected to contain a negative baseline"),
    )


class DemixJobSchema(ArgSchema):
    movie_h5 = H5InputFile(
        required=True,
        description=("Input h5 file containing the motion corrected movie "),
    )
    traces_h5 = H5InputFile(
        required=True,
        description=(
            "Input h5 file containing fluorescence traces and their "
            "associated ROI IDs (in datasets specified by the keys "
            "'data' and 'roi_names', respectively."
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
    exclude_labels = fields.List(
        fields.Str,
        required=False,
        many=True,
        default=EXCLUDE_LABELS,
        description=(
            "List of exlusion labels that will invalidate an ROI if an ROI"
            "is tagged with an exclusion_label from this list."
        ),
    )
