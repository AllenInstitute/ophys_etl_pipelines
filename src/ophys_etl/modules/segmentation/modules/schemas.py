import h5py
import argschema
from marshmallow import pre_load, post_load, ValidationError
from marshmallow.fields import Int
from marshmallow.validate import OneOf


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


class CalculateEdgesInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    graph_input = argschema.fields.InputFile(
        required=False,
        description=("read by nx.read_gpickle() for graph input. If "
                     "not provided, graph will be created from video "
                     "shape"))
    create_graph_args = argschema.fields.Nested(
        CreateGraphInputSchema,
        required=False,
        default={},
        description=("if 'graph_input' not provided, the graph will be "
                     "created from these args."))
    video_path = argschema.fields.InputFile(
        required=True,
        description=("path to hdf5 video with movie stored "
                     "in dataset 'data' nframes x nrow x ncol"))
    graph_output = argschema.fields.OutputFile(
        required=True,
        description="written by nx.write_gpickle() for graph output")
    plot_output = argschema.fields.OutputFile(
        required=False,
        description=("if provided, will create a plot saved to this location.",
                     "The format is inferred from the extension by "
                     "matplotlib.figure.Figure.savefig()"))
    attribute = argschema.fields.Str(
        required=False,
        default="Pearson",
        validate=OneOf(["Pearson", "filtered_Pearson", "hnc_Gaussian",
                        "filtered_hnc_Gaussian"]),
        description="which calculation to perform")
    filter_fraction = argschema.fields.Float(
        required=False,
        default=0.2,
        description="Fraction of timesteps to kee if "
                    "calculating the filtered Pearson coefficient")
    neighborhood_radius = argschema.fields.Int(
        required=False,
        default=15,
        description=("size of neighborhood radius (in pixels) for "
                     "hnc gaussian distance."))
    full_neighborhood = argschema.fields.Bool(
        required=False,
        default=False,
        description = ("if True, use the full neighborhood when "
                       "selecting timesteps to keep in filtered_hnc_* "
                       "graphs (default=False)"))
    n_parallel_workers = argschema.fields.Int(
        required=False,
        default=1,
        description=("how many multiprocessing workers to use. If set to "
                     "1, multiprocessing is not invoked."))

    @pre_load
    def set_create_graph_args(self, data, **kwargs):
        for k in ["video_path", "graph_output"]:
            data["create_graph_args"][k] = data[k]
        return data


class GraphPlotInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    graph_input = argschema.fields.InputFile(
        required=True,
        description="source file for networkx.read_gpickle()")
    plot_output = argschema.fields.OutputFile(
        required=True,
        description=("destination png for plot"))


class SegmentV0InputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    graph_input = argschema.fields.InputFile(
        required=True,
        description="source file for networkx.read_gpickle()")
    n_partitions = argschema.fields.Int(
        required=False,
        default=1,
        description=("how many kmeans sub-graphs to create and send "
                     "to distinct workers. Can accelerate by parallelizing "
                     "and by reducing the graph size for any one worker. If "
                     "1, multiprocessing is not invoked."))
    attribute_name = argschema.fields.Str(
        required=False,
        default="Pearson",
        description="which edge attribute name to operate on")
    seed_quantile = argschema.fields.Float(
        required=False,
        default=0.95,
        description=("starting seeds for graph growth are determined "
                     "by the connected components that remain after "
                     "keeping only the edges above this quantile. "
                     "applied iteratively as ROIs are detected and "
                     "removed from graph."))
    graph_output = argschema.fields.OutputFile(
        required=True,
        description="written by nx.write_gpickle() for graph output")
    plot_output = argschema.fields.OutputFile(
        required=False,
        description="if provided, will create a before/after plot")
    roi_output = argschema.fields.OutputFile(
        required=False,
        description="if provided, will write subgraphs to json as ROIs")


class SimpleDenoiseInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    video_path = argschema.fields.InputFile(
        required=True,
        description=("path to hdf5 video with movie stored "
                     "in dataset 'data' nframes x nrow x ncol"))
    video_output = argschema.fields.OutputFile(
        required=True,
        description="destination path to filtered hdf5 video ")
    size = argschema.fields.Float(
        required=True,
        description=("filter size for the time axis. "
                     "If filter_type is 'uniform' this value will be cast "
                     "to an integer and used as a boxcar width. If "
                     "filter_type is 'gaussian', this value remains a float "
                     "and is the sigma for the Gaussian filter."))
    filter_type = argschema.fields.Str(
        required=True,
        validate=OneOf(["uniform", "gaussian"]),
        description=("the type of temporal filter to apply to each pixel's "
                     "trace."))
    n_parallel_workers = argschema.fields.Int(
        required=False,
        default=1,
        description=("how many multiprocessing workers to use. If set to "
                     "1, multiprocessing is not invoked."))
