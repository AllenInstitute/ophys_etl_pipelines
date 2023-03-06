from argschema import ArgSchema, fields

from ophys_etl.schemas.fields import H5InputFile


class NeuropilCorrectionJobSchema(ArgSchema):
    neuropil_trace_file = H5InputFile(
        required=True,
        description=("Path to input h5 file containing neuropil traces"),
    )

    roi_trace_file = H5InputFile(
        required=True,
        description=("Path to input h5 file containing roi traces"),
    )

    storage_directory = fields.OutputDir(
        required=True, description=("Path to output directory")
    )

    motion_corrected_stack = H5InputFile(
        required=True,
        description=("Path to input h5 file containing neuropil traces"),
    )
