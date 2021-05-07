import argschema
import itertools
import networkx as nx
import numpy as np
from typing import List, Tuple, Optional

from ophys_etl.modules.segmentation.modules.create_graph.schemas \
    import CreateGraphInputSchema


def create_graph(row_min: int, row_max: int, col_min: int, col_max: int,
                 kernel: Optional[List[Tuple[int, int]]] = None) -> nx.Graph:
    """produces a graph with nodes densely packing the inclusive
    bounds defined by row/col_min/max

    Parameters
    ----------
    row_min: int
        minimum value of row index
    row_max: int
        maximum value of row index
    col_min: int
        minimum value of col index
    col_max: int
        maximum value of col index
    kernel: List[Tuple[int, int]]
        N x 2: [(r0, c0), (r1, c1), ...] each (ri, ci) pair
        defines a relative (row, col) neighbor for creating a graph edge

    Returns
    -------
    graph: nx.Graph
        an undirected networkx graph, free of attributes

    """

    if kernel is None:
        # relative indices of 8 nearest-neighbors
        kernel = list(itertools.product([-1, 0, 1], repeat=2))
        kernel.pop(kernel.index((0, 0)))

    rows, cols = np.mgrid[row_min:(row_max + 1), col_min:(col_max + 1)]
    graph = nx.Graph()
    for edge_start in zip(rows.flat, cols.flat):
        for drow, dcol in kernel:
            edge_end = (edge_start[0] + drow, edge_start[1] + dcol)
            if (edge_end[0] < row_min) | (edge_end[0] > row_max):
                continue
            if (edge_end[1] < col_min) | (edge_end[1] > col_max):
                continue
            edge = [edge_start, edge_end]
            if not graph.has_edge(*edge):
                graph.add_edge(*edge)
    return graph


class CreateGraph(argschema.ArgSchemaParser):
    default_schema = CreateGraphInputSchema

    def run(self):
        self.logger.name = type(self).__name__
        graph = create_graph(
                    row_min=self.args["row_min"],
                    row_max=self.args["row_max"],
                    col_min=self.args["col_min"],
                    col_max=self.args["col_max"],
                    kernel=self.args["kernel"])
        nx.write_gpickle(graph, self.args["graph_output"])
        self.logger.info(f"wrote {self.args['graph_output']}")


if __name__ == "__main__":  # pragma: nocover
    cg = CreateGraph()
    cg.run()
