import h5py
import argschema
from marshmallow import post_load, ValidationError
from marshmallow.fields import Int


class CreateGraphInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    video_path = argschema.fields.InputFile(
        required=False,
        description=("path to hdf5 video with movie stored "
                     "in dataset 'data' nframes x nrow x ncol"))
    row_min = argschema.fields.Int(
        required=False,
        description="minimum row index for nodes")
    row_max = argschema.fields.Int(
        required=False,
        description="maximum row index for nodes")
    col_min = argschema.fields.Int(
        required=False,
        description="minimum column index for nodes")
    col_max = argschema.fields.Int(
        required=False,
        description="maximum column index for nodes")
    kernel = argschema.fields.List(
        argschema.fields.Tuple((Int(), Int())),
        cli_as_single_argument=True,
        required=False,
        allow_none=True,
        default=None,
        description=("list of (row, col) entries that define the "
                     "relative location of nodes for establishing edges."))
    graph_output = argschema.fields.OutputFile(
        required=True,
        description="destination file for networkx.write_gpickle()")

    @post_load
    def set_row_col(self, data, **kwargs):
        rowcol = [i in data for i in ["row_min", "row_max",
                                      "col_min", "col_max"]]
        if (not all(rowcol)) & ("video_path" not in data):
            raise ValidationError("provide either all 4 of row/col_min/max "
                                  "or a valid video_path")
        if "video_path" in data:
            with h5py.File(data["video_path"], "r") as f:
                nrow, ncol = f["data"].shape[1:]
            data["row_min"] = 0
            data["row_max"] = nrow - 1
            data["col_min"] = 0
            data["col_max"] = ncol - 1
        return data
