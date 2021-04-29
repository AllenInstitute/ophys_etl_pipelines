import argschema

from ophys_etl.qc.video.schemas import GraphCreationSchema


class PearsonSegmentationInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    graph_creation = argschema.fields.Nested(
        GraphCreationSchema)
    graph_input = argschema.fields.InputFile(
        required=False,
        description=("source file for networkx.read_gpickle(). If "
                     "not provided, will be created. "
                     "must provide 'graph_input' or 'graph_creation' args"))
    normalize = argschema.fields.Boolean(
        required=False,
        default=True,
        description=("whether to normalize the graph edge weights to a local "
                     "Gaussian filter."))
    sigma = argschema.fields.Float(
        required=False,
        default=30.0,
        description="size of Gaussian for local edge weight normalization.")
    mask_quantile = argschema.fields.Float(
        required=False,
        default=0.5,
        description=("Pixels whose stdev/mean edge weights that exceed "
                     "this quantile will be masked to aid the watershed "
                     "segmentation algorithm."))
    roi_output = argschema.fields.OutputFile(
        required=True,
        description="LIMS format ROI file")
