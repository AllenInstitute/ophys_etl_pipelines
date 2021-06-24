import argschema
import networkx as nx
from matplotlib import figure

from ophys_etl.modules.segmentation.qc import graph_plotting
from ophys_etl.modules.segmentation.modules.schemas import \
    GraphPlotInputSchema


class GraphPlot(argschema.ArgSchemaParser):
    default_schema = GraphPlotInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        graph = nx.read_gpickle(self.args["graph_input"])
        fig = figure.Figure(figsize=(16, 16))
        axis = fig.add_subplot(111)
        if self.args['draw_edges']:
            graph_plotting.draw_graph_edges(
                    fig,
                    axis,
                    graph,
                    attribute_name=self.args['attribute_name'])
        else:
            graph_plotting.draw_graph_image(
                    fig,
                    axis,
                    graph,
                    attribute_name=self.args['attribute_name'])
        fig.savefig(self.args["plot_output"], dpi=300)
        self.logger.info(f"wrote {self.args['plot_output']}")


if __name__ == "__main__":  # pragma: nocover
    cg = GraphPlot()
    cg.run()
