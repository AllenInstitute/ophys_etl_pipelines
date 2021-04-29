import numpy as np
import networkx as nx
from scipy.ndimage import gaussian_filter


def normalize_graph(graph, sigma=30):
    """normalizes edge weights by a local Gaussian filter
    of size sigma.
    """
    coords = np.array(list(graph.nodes))
    shape = tuple(coords.max(axis=0) + 1)
    avgw = np.zeros(shape, dtype='float')
    for node in graph.nodes:
        n = len(list(graph.neighbors(node)))
        avgw[node[0], node[1]] = graph.degree(node, weight="weight") / n
    avgw = gaussian_filter(avgw, sigma, mode='nearest')
    weights = []
    for i, v in graph.edges.items():
        local = 0.5 * (avgw[i[0][0], i[0][1]] + avgw[i[1][0], i[1][0]])
        weights.append(v["weight"] / local)
    c = nx.Graph()
    for i, e in enumerate(graph.edges()):
        c.add_edge(*e, weight=weights[i])
    return c
