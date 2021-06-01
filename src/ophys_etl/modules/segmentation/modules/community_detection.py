import argschema
import multiprocessing
import tempfile
import json
import networkx as nx
from matplotlib.figure import Figure
from pathlib import Path

from ophys_etl.modules.segmentation.graph_utils import (partition,
                                                        community,
                                                        plotting)
from ophys_etl.modules.segmentation.modules.schemas import \
    SegmentV0InputSchema


class SegmentV0(argschema.ArgSchemaParser):
    default_schema = SegmentV0InputSchema

    def run(self):
        self.logger.name = type(self).__name__

        graph = nx.read_gpickle(self.args["graph_input"])
        if self.args["n_partitions"] == 1:
            new_graph = community.iterative_detection(
                    graph,
                    self.args["attribute_name"],
                    self.args["seed_quantile"])
        else:
            subgraphs = partition.partition_graph_by_edges(
                    graph, self.args["n_partitions"])

            args = []
            with tempfile.TemporaryDirectory() as tdir:
                for i, subgraph in enumerate(subgraphs):
                    gpath = Path(tdir) / f"{i}.pkl"
                    nx.write_gpickle(subgraph, gpath)
                    args.append((gpath,
                                 self.args["attribute_name"],
                                 self.args["seed_quantile"]))
                with multiprocessing.Pool(len(subgraphs)) as pool:
                    results = pool.starmap(community.iterative_detection,
                                           args)
                new_graph = nx.compose_all([nx.read_gpickle(i)
                                            for i in results
                                            if i is not None])

        nx.write_gpickle(new_graph, self.args["graph_output"])
        self.logger.info(f"wrote {self.args['graph_output']}")

        if 'plot_output' in self.args:
            fig = Figure(figsize=(18, 9), dpi=300)
            a0 = fig.add_subplot(121)
            a1 = fig.add_subplot(122, sharex=a0, sharey=a0)
            plotting.draw_graph_edges(fig, a0, graph,
                                      self.args["attribute_name"])
            plotting.draw_graph_edges(fig, a1, new_graph,
                                      self.args["attribute_name"])
            fig.savefig(self.args["plot_output"])
            self.logger.info(f"wrote {self.args['plot_output']}")

        if 'roi_output' in self.args:
            subgraphs = [new_graph.subgraph(i)
                         for i in nx.connected_components(new_graph)]
            rois = [community.graph_to_roi(subgraph, i)
                    for i, subgraph in enumerate(subgraphs)]
            with open(self.args["roi_output"], "w") as f:
                json.dump(rois, f, indent=2)
            self.logger.info(f"wrote {self.args['roi_output']}")


if __name__ == "__main__":
    seg = SegmentV0()
    seg.run()
