import argschema


class GraphSegmentationInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    graph_input = argschema.fields.InputFile(
        required=True,
        description="source file for networkx.read_gpickle()")
    louvain_resolution = argschema.fields.Float(
        required=True,
        description="passed to sknetwork.clustering.Louvain(resolution=)")
    roi_output = argschema.fields.OutputFile(
        required=True,
        description="LIMS format ROI file")
