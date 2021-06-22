import argschema
import time
import networkx as nx
import multiprocessing
import tempfile
import h5py
from typing import Optional
from pathlib import Path
from functools import partial
import matplotlib

from ophys_etl.modules.segmentation.modules.schemas import \
    CalculateEdgesInputSchema
from ophys_etl.modules.segmentation.graph_utils import (
        partition, edge_attributes, creation)
from ophys_etl.modules.segmentation.qc import graph_plotting


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
        filter_fraction: Optional[float] = None,
        full_neighborhood: bool = False) -> nx.Graph:
    graph = edge_attributes.add_hnc_gaussian_metric(
            graph=nx.read_gpickle(graph_path),
            video_path=video_path,
            neighborhood_radius=neighborhood_radius,
            filter_fraction=filter_fraction,
            attribute_name=attribute_name,
            full_neighborhood=full_neighborhood)
    nx.write_gpickle(graph, graph_path)
    return graph_path


class CalculateEdges(argschema.ArgSchemaParser):
    default_schema = CalculateEdgesInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        t0 = time.time()

        # select which calculation to perform
        if self.args['attribute_name'] == 'Pearson':
            edge_job = pearson_edge_job
        elif self.args['attribute_name'] == 'filtered_Pearson':
            edge_job = partial(filtered_pearson_edge_job,
                               self.args['filter_fraction'])
        elif self.args['attribute_name'] == 'hnc_Gaussian':
            edge_job = partial(hnc_gaussian_edge_job,
                               self.args['neighborhood_radius'],
                               attribute_name=self.args['attribute_name'])
        elif self.args['attribute_name'] == 'filtered_hnc_Gaussian':
            edge_job = partial(
                          hnc_gaussian_edge_job,
                          self.args['neighborhood_radius'],
                          filter_fraction=self.args['filter_fraction'],
                          attribute_name=self.args['attribute_name'],
                          full_neighborhood=self.args['full_neighborhood'])

        if "graph_input" not in self.args:
            with h5py.File(self.args["video_path"], "r") as f:
                shape = f["data"].shape[1:]
            graph = creation.create_graph(0, shape[0] - 1, 0, shape[1] - 1,
                                          self.args["kernel"])
            nx.write_gpickle(graph, self.args["graph_output"])
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
            graph_plotting.draw_graph_edges(
                    fig, axes, graph, self.args["attribute_name"])
            fig.savefig(self.args["plot_output"])
            self.logger.info(f"wrote {self.args['plot_output']}")

        self.logger.info(f"finished in {time.time() - t0:2f} seconds")


if __name__ == "__main__":  # pragma: nocover
    ce = CalculateEdges()
    ce.run()
