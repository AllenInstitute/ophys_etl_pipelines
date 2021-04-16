import argschema


class CorrelationGraphInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    video_path = argschema.fields.InputFile(
        required=True,
        description=("path to hdf5 video with movie stored "
                     "in dataset 'data' nframes x nrow x ncol"))
    n_segments = argschema.fields.Int(
        required=False,
        default=1,
        description=("number of segments in both x and y to engage "
                     "multiprocessing. n_segments^2 workers will be "
                     "created to distribute the load. If == 1, "
                     "multiprocessing not invoked at all."))
    graph_output = argschema.fields.OutputFile(
        required=True,
        description="destination file for networkx.write_gpickle()")
    plot_output = argschema.fields.OutputFile(
        required=False,
        description=("if provided, will create a plot and write to this "
                     "location. passed to matplotlib.pyplot.Figure.savefig()"
                     ".png being a typical suffix"))


class CorrelationGraphPlotInputSchema(argschema.ArgSchema):
    log_level = argschema.fields.LogLevel(default="INFO")
    graph_input = argschema.fields.InputFile(
        required=True,
        description="source file for networkx.read_gpickle()")
    plot_output = argschema.fields.OutputFile(
        required=True,
        description=("destination png for plot"))
