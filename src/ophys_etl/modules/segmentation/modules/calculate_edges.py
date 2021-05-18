import argschema
import time
import networkx as nx
import multiprocessing
import tempfile
from typing import Optional
from pathlib import Path
from functools import partial
import matplotlib

from ophys_etl.modules.segmentation.modules.create_graph import CreateGraph
from ophys_etl.modules.segmentation.modules.schemas import \
    CalculateEdgesInputSchema
from ophys_etl.modules.segmentation.graph_utils import (
    partition, edge_attributes, plotting)

matplotlib.use("agg")


def pearson_edge_job(graph_path: Path, video_path: Path) -> nx.Graph:
    graph = edge_attributes.add_pearson_edge_attributes(
            nx.read_gpickle(graph_path),
            video_path)
    nx.write_gpickle(graph, graph_path)
    return graph_path


def filtered_pearson_edge_job(filter_fraction: float,
                              graph_path: Path,
                              video_path: Path) -> nx.Graph:
    graph = edge_attributes.add_filtered_pearson_edge_attributes(
            filter_fraction,
            nx.read_gpickle(graph_path),
            video_path)
    nx.write_gpickle(graph, graph_path)
    return graph_path


def hnc_gaussian_edge_job(
        neighborhood_radius: int,
        graph_path: Path,
        video_path: Path,
        attribute_name: str = 'hnc_Gaussian',
        filter_fraction: Optional[float] = None) -> nx.Graph:
    graph = edge_attributes.add_hnc_gaussian_metric(
            graph=nx.read_gpickle(graph_path),
            video_path=video_path,
            neighborhood_radius=neighborhood_radius,
            filter_fraction=filter_fraction,
            attribute_name=attribute_name)
    nx.write_gpickle(graph, graph_path)
    return graph_path


class CalculateEdges(argschema.ArgSchemaParser):
    default_schema = CalculateEdgesInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        t0 = time.time()

        if self.args['attribute'] == 'Pearson':
            edge_job = pearson_edge_job
        elif self.args['attribute'] == 'filtered_Pearson':
            edge_job = partial(filtered_pearson_edge_job,
                               self.args['filter_fraction'])
        elif self.args['attribute'] == 'hnc_Gaussian':
            edge_job = partial(hnc_gaussian_edge_job,
                               self.args['neighborhood_radius'],
                               attribute_name=self.args['attribute'])
        elif self.args['attribute'] == 'filtered_hnc_Gaussian':
            edge_job = partial(hnc_gaussian_edge_job,
                               self.args['neighborhood_radius'],
                               filter_fraction=self.args['filter_fraction'],
                               attribute_name=self.args['attribute'])

        if "graph_input" not in self.args:
            cg = CreateGraph(input_data=self.args["create_graph_args"],
                             args=[])
            cg.run()
            self.args["graph_input"] = self.args["graph_output"]
            self.logger.name = type(self).__name__

        if self.args["n_parallel_workers"] == 1:
            graph_path = edge_job(self.args["graph_input"],
                                  self.args["video_path"])
            graph = nx.read_gpickle(graph_path)
        else:
            subgraphs = partition.partition_graph_by_edges(
                    nx.read_gpickle(self.args["graph_input"]),
                    self.args["n_parallel_workers"])
            with tempfile.TemporaryDirectory() as tdir:
                args = []
                for i, subgraph in enumerate(subgraphs):
                    gpath = str(Path(tdir) / f"{i}.pkl")
                    nx.write_gpickle(subgraph, gpath)
                    args.append((gpath, self.args["video_path"]))
                with multiprocessing.Pool(
                        self.args["n_parallel_workers"]) as pool:
                    results = pool.starmap(edge_job, args)
                graph = nx.compose_all([nx.read_gpickle(i) for i in results])

        nx.write_gpickle(graph, self.args["graph_output"])
        self.logger.info(f"wrote {self.args['graph_output']}")

        if "plot_output" in self.args:
            fig = matplotlib.figure.Figure(figsize=(16, 16), dpi=300)
            axes = fig.add_subplot(111)
            plotting.draw_graph_edges(fig, axes, graph, self.args["attribute"])
            fig.savefig(self.args["plot_output"])
            self.logger.info(f"wrote {self.args['plot_output']}")

        self.logger.info(f"finished in {time.time() - t0:2f} seconds")


if __name__ == "__main__":
    ce = CalculateEdges()
    ce.run()
