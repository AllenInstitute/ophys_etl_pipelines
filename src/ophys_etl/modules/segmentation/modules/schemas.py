import argschema
import h5py
import numpy as np
from marshmallow import post_load, ValidationError
from marshmallow.fields import Int
from marshmallow.validate import OneOf


class CalculateEdgesInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    graph_input = argschema.fields.InputFile(
        required=False,
        description=("read by nx.read_gpickle() for graph input. If "
                     "not provided, graph will be created from video "
                     "shape"))
    video_path = argschema.fields.InputFile(
        required=True,
        description=("path to hdf5 video with movie stored "
                     "in dataset 'data' nframes x nrow x ncol"))
    graph_output = argschema.fields.OutputFile(
        required=True,
        description="written by nx.write_gpickle() for graph output")
    plot_output = argschema.fields.OutputFile(
        required=False,
        description=("if provided, will create a plot saved to this location."
                     "The format is inferred from the extension by "
                     "matplotlib.figure.Figure.savefig()"))
    attribute_name = argschema.fields.Str(
        required=False,
        default="Pearson",
        validate=OneOf(["Pearson", "filtered_Pearson", "hnc_Gaussian",
                        "filtered_hnc_Gaussian"]),
        description="which calculation to perform")
    filter_fraction = argschema.fields.Float(
        required=False,
        default=0.2,
        validate=lambda x: 0.0 < x <= 1.0,
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
        description=("if True, use the full neighborhood when "
                     "selecting timesteps to keep in filtered_hnc_* "
                     "graphs (default=False)"))
    n_parallel_workers = argschema.fields.Int(
        required=False,
        default=1,
        description=("how many multiprocessing workers to use. If set to "
                     "1, multiprocessing is not invoked."))
    kernel = argschema.fields.List(
        argschema.fields.Tuple(
            (argschema.fields.Int(),
             argschema.fields.Int())),
        cli_as_single_argument=True,
        required=False,
        allow_none=True,
        default=None,
        description=("list of (row, col) entries that define the "
                     "relative location of nodes for establishing edges. "
                     "If left as None, an 8 nearest-neighbor kernel "
                     "will be used."))


class GraphPlotInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    graph_input = argschema.fields.InputFile(
        required=True,
        description="source file for networkx.read_gpickle()")
    plot_output = argschema.fields.OutputFile(
        required=True,
        description=("destination png for plot"))
    draw_edges = argschema.fields.Boolean(
        required=False,
        default=False,
        description=("If true, draw edges of graph. "
                     "If false, draw graph as pixel image "
                     "in which a pixel's intensity is the sum of "
                     "the edge weights connected to that pixel. "))
    attribute_name = argschema.fields.Str(
        required=False,
        default=None,
        allow_none=True,
        description=("which attribute to use in image. If None, will search "
                     "for a unique edge attribute name, or raise an "
                     "exception if there is not one and only one name "
                     "available."))


class DenoiseBaseSchema(argschema.schemas.DefaultSchema):
    video_path = argschema.fields.InputFile(
        required=True,
        description=("path to hdf5 video with movie stored "
                     "in dataset 'data' nframes x nrow x ncol"))
    video_output = argschema.fields.OutputFile(
        required=True,
        description="destination path to filtered hdf5 video ")
    h5_chunk_shape = argschema.fields.Tuple(
        (Int(), Int(), (Int())),
        default=(50, 32, 32),
        description="passed to h5py.File.create_dataset(chunks=)")


class PCADenoiseInputSchema(argschema.ArgSchema, DenoiseBaseSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    n_components = argschema.fields.Int(
        required=True,
        description=("number of principal components to keep. "
                     "the chunking of the movie for incremental PCA requires "
                     "that there are more than 'n_components' frames in each "
                     "chunk. See 'n_chunks' parameter."))
    n_chunks = argschema.fields.Int(
        required=True,
        description=("the number of temporal chunks to send iteratively to "
                     "IncrementalPCA.partial_fit(). A smaller number will "
                     "have a larger memory footprint."))

    @post_load
    def check_chunking(self, data, **kwargs):
        with h5py.File(data["video_path"], "r") as f:
            nframes = f["data"].shape[0]
        ind_split = np.array_split(np.arange(nframes), data["n_chunks"])
        min_size = min([i.size for i in ind_split])
        if min_size < data["n_components"]:
            raise ValidationError(f"the input movie has {nframes} frames "
                                  f"and when split into {data['n_chunks']} "
                                  f"chunks, the smallest chunk is {min_size} "
                                  "frames in size. The chunks can not be "
                                  "smaller than the number of components: "
                                  f"{data['n_components']}. Decrease either "
                                  "'n_components' or 'n_chunks'")
        return data


class SimpleDenoiseInputSchema(argschema.ArgSchema, DenoiseBaseSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
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
