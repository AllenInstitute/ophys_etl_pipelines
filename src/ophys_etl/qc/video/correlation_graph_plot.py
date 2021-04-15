import argschema
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ophys_etl.qc.video.schemas import CorrelationGraphPlotInputSchema


def draw_graph_edges(figure, axis, graph):

    weights = np.array([i["weight"] for i in graph.edges.values()])
    # graph is (row, col), transpose to get (x, y)
    segments = []
    for edge in graph.edges:
        segments.append([edge[0][::-1], edge[1][::-1]])
    line_coll = LineCollection(segments, linestyle='solid',
                               cmap="plasma")
    line_coll.set_array(weights)
    axis.add_collection(line_coll)
    vals = np.concatenate(line_coll.get_segments())
    mnvals = vals.min(axis=0)
    mxvals = vals.max(axis=0)
    ppvals = vals.ptp(axis=0)
    buffx = 0.02 * ppvals[0]
    buffy = 0.02 * ppvals[1]
    axis.set_xlim(mnvals[0] - buffx, mxvals[0] + buffx)
    axis.set_ylim(mnvals[1] - buffy, mxvals[1] + buffy)
    axis.set_aspect("equal")
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    figure.colorbar(line_coll, ax=axis, cax=cax)
    axis.set_title("PCC neighbor graph")


class CorrelationGraphPlot(argschema.ArgSchemaParser):
    default_schema = CorrelationGraphPlotInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        graph = nx.read_gpickle(self.args["graph_input"])
        fig, axis = plt.subplots(1, 1, clear=True, num=1, figsize=(8, 8))
        draw_graph_edges(fig, axis, graph)

        fig.savefig(self.args["plot_output"])
        self.logger.info(f"wrote {self.args['plot_output']}")


if __name__ == "__main__":
    cg = CorrelationGraphPlot()
    cg.run()
