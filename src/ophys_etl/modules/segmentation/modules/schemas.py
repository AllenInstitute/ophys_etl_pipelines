import h5py
import argschema
import numpy as np
from pathlib import Path
from marshmallow import pre_load, post_load, ValidationError
from marshmallow.fields import Int
from marshmallow.validate import OneOf

from ophys_etl.modules.segmentation.seed.schemas import \
    ImageMetricSeederSchema, BatchImageMetricSeederSchema


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
        description=("if provided, will create a plot saved to this location."
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
        description=("if True, use the full neighborhood when "
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
    draw_edges = argschema.fields.Boolean(
        required=False,
        default=False,
        description=("If true, draw edges of graph. "
                     "If false, draw graph as pixel image "
                     "in which a pixel's intensity is the sum of "
                     "the edge weights connected to that pixel. "))
    attribute = argschema.fields.Str(
        required=False,
        default="Pearson",
        validate=OneOf(["Pearson", "filtered_Pearson", "hnc_Gaussian",
                        "filtered_hnc_Gaussian"]),
        description="which attribute to use in image")


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


class FeatureVectorSegmentationInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")

    seeder_args = argschema.fields.Nested(
        BatchImageMetricSeederSchema,
        default={})

    video_input = argschema.fields.InputFile(
        required=False,
        description=("path to hdf5 video with movie stored "
                     "in dataset 'data' nframes x nrow x ncol"))

    graph_input = argschema.fields.InputFile(
        required=False,
        description=("path to graph used to seed ROIS"))

    attribute = argschema.fields.Str(
        required=False,
        default="Pearson",
        validate=OneOf(["Pearson", "filtered_Pearson", "hnc_Gaussian",
                        "filtered_hnc_Gaussian"]),
        description="which attribute to use in image")

    roi_output = argschema.fields.OutputFile(
        required=True,
        description="path to json file where ROIs will be saved")

    plot_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="path to summary plot of segmentation")

    qc_output = argschema.fields.OutputFile(
        required=True,
        default=None,
        description=("path to hdf5 QC output"))

    seed_plot_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description=("path to plot of seeding summary."))

    n_parallel_workers = argschema.fields.Int(
        required=False,
        default=1,
        description=("how many multiprocessing workers to use."))

    roi_class = argschema.fields.Str(
        required=False,
        default="PearsonFeatureROI",
        validate=OneOf(["PearsonFeatureROI", "PCAFeatureROI"]),
        description="which class to use.")

    filter_fraction = argschema.fields.Float(
        required=False,
        default=0.2,
        description="fraction of timesteps to use in time correlation")

    min_window_size = argschema.fields.Int(
        required=False,
        default=20,
        description=("minimum size of window an ROI will be given to "
                     "grow in"))

    max_window_size = argschema.fields.Int(
        required=False,
        default=40,
        description=("maximum size of window an ROI will be given to "
                     "grow in"))

    @post_load
    def plot_outputs(self, data, **kwargs):
        if data['seed_plot_output'] is None:
            if data['plot_output'] is not None:
                plot_path = Path(data['plot_output'])
            data['seed_plot_output'] = str(
                    plot_path.parent / f"{plot_path.stem}_seeds.png")
        return data


class HNC_args(argschema.schemas.DefaultSchema):
    """
    see
    https://github.com/hochbaumGroup/HNCcorr/blob/764e45ae3976fbc2519c75cd13b7f7b22c6a38dc/src/hnccorr/base.py#L347-L362  # noqa: E501
    and
    https://hnccorr.readthedocs.io/en/latest/quickstart.html#configuration  # noqa: E501
    """
    postprocessor_min_cell_size = argschema.fields.Int(
        default=40,
        description="Lower bound on pixel count of a cell.")
    postprocessor_preferred_cell_size = argschema.fields.Int(
        default=80,
        description="Pixel count of a typical cell.")
    postprocessor_max_cell_size = argschema.fields.Int(
        default=200,
        description="Upper bound on pixel count of a cell.")
    patch_size = argschema.fields.Int(
        default=31,
        description="Size in pixel of each dimension of the patch.")
    positive_seed_radius = argschema.fields.Int(
        default=0,
        description="Radius of the positive seed square / superpixel.")
    negative_seed_circle_radius = argschema.fields.Int(
        default=10,
        description="Radius in pixels of the circle with negative seeds.")
    negative_seed_circle_count = argschema.fields.Int(
        default=10,
        description="Number of negative seeds.")
    gaussian_similarity_alpha = argschema.fields.Float(
        default=1,
        description="Decay factor in gaussian similarity function.")
    sparse_computation_grid_distance = argschema.fields.Float(
        default=1 / 35.0,
        description=("1 / grid_resolution. Width of each block in "
                     "sparse computation."))
    sparse_computation_dimension = argschema.fields.Int(
        default=3,
        description=("Dimension of the low-dimensional space in sparse "
                     "computation."))


class HNCSegmentationWrapperInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    video_input = argschema.fields.InputFile(
        required=False,
        description=("path to hdf5 video with movie stored "
                     "in dataset 'data' nframes x nrow x ncol"))
    graph_input = argschema.fields.InputFile(
        required=True,
        description=("path to graph used to seed ROIS"))
    attribute_name = argschema.fields.Str(
        required=False,
        default='filtered_hnc_Gaussian',
        validate=OneOf(["Pearson", "filtered_Pearson", "hnc_Gaussian",
                        "filtered_hnc_Gaussian"]),
        description="which graph edge attribute to use to create image.")
    experiment_name = argschema.fields.Str(
        required=False,
        default="movie_name",
        description="passed to HNCcorr.Movie as 'name'")
    seeder_args = argschema.fields.Nested(
        ImageMetricSeederSchema,
        default={})
    hnc_args = argschema.fields.Nested(HNC_args, default={})
    roi_output = argschema.fields.OutputFile(
        required=True,
        description="path to json file where ROIs will be saved")
    seed_output = argschema.fields.OutputFile(
        required=True,
        description="path to json file where seeds will be saved")
    plot_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="path to summary plot of segmentation")
    seed_plot_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description=("path to plot of seeding summary."))

    @post_load
    def plot_outputs(self, data, **kwargs):
        if data['seed_plot_output'] is None:
            if data['plot_output'] is not None:
                plot_path = Path(data['plot_output'])
            data['seed_plot_output'] = str(
                    plot_path.parent / f"{plot_path.stem}_seeds.png")
        return data


class RoiMergerSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")

    roi_input = argschema.fields.InputFile(
        required=True,
        description=("path to JSON file with ROIs to merge"))

    qc_output = argschema.fields.OutputFile(
        required=True,
        default=None,
        description=("path to hdf5 QC output"))

    plot_output = argschema.fields.OutputFile(
        required=False,
        default=None,
        allow_none=True,
        description="path to summary plot of segmentation")

    roi_output = argschema.fields.OutputFile(
        required=True,
        description=("path to JSON file where we will write merged ROIs"))

    n_parallel_workers = argschema.fields.Int(
            required=False,
            default=8,
            description=("number of parallel processes to use"))

    attribute = argschema.fields.Str(
        required=False,
        default="filtered_hnc_Gaussian",
        validate=OneOf(["Pearson", "filtered_Pearson", "hnc_Gaussian",
                        "filtered_hnc_Gaussian"]),
        description="which attribute to use in image")

    video_input = argschema.fields.InputFile(
        required=True,
        description=("path to hdf5 video with movie stored "
                     "in dataset 'data' nframes x nrow x ncol"))

    corr_acceptance = argschema.fields.Float(
        required=False,
        default=2.0,
        decription=("level of time series correlation needed "
                    "to accept a merger (in units of z-score)"))

    anomalous_size = argschema.fields.Int(
        required=False,
        default=800,
        description=("If an ROI reaches this size, it is considered "
                     "invalid and removed from the merging process"))

    filter_fraction = argschema.fields.Float(
        required=False,
        default=0.2,
        description=("fraction of timesteps to keep when doing time "
                     "correlations"))
