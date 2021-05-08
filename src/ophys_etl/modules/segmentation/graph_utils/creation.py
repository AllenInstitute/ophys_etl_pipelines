import itertools
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional


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

    graph = nx.Graph()

    if (row_min == row_max) & (col_min == col_max):
        # trivial case with no edges and 1 node
        graph.add_node((row_min, col_min))
        return graph

    rows, cols = np.mgrid[row_min:(row_max + 1), col_min:(col_max + 1)]
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
