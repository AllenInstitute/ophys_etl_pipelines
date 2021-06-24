import h5py
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple, List
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter


def normalize_graph(graph: nx.Graph,
                    attribute_name: str,
                    sigma: float = 30.0,
                    new_attribute_name: Optional[str] = None) -> nx.Graph:
    """normalizes edge weights by a local Gaussian filter
    of size sigma.

    Parameters
    ----------
    graph: nx.Graph
        a graph with edge weights
    sigma: float
        passed to scipy.ndimage.gaussian_filter as 'sigma'
    attribute_name: str
        the name of the edge attribute to normalize
    new_attribute_name: str
        the name of the new edge attribute. If 'None' will be
        <attribute_name>_normalized

    Returns
    -------
    graph: nx.Graph
        graph with the new edge attribute

    """
    if new_attribute_name is None:
        new_attribute_name = attribute_name + "_normalized"

    # create an image that is the average edge attribute per node
    coords = np.array(list(graph.nodes))
    shape = tuple(coords.max(axis=0) + 1)
    avg_attr = np.zeros(shape, dtype='float')
    for node in graph.nodes:
        n = len(list(graph.neighbors(node)))
        avg_attr[node[0], node[1]] = \
            graph.degree(node, weight=attribute_name) / n

    # Gaussian filter the average image
    avg_attr = gaussian_filter(avg_attr, sigma, mode='nearest')
    edge_values = dict()
    for i, v in graph.edges.items():
        local = 0.5 * (avg_attr[i[0][0], i[0][1]] + avg_attr[i[1][0], i[1][0]])
        edge_values[i] = {new_attribute_name: v[attribute_name] / local}

    nx.set_edge_attributes(G=graph, values=edge_values)
    return graph


def add_pearson_edge_attributes(graph: nx.Graph,
                                video_path: Path,
                                attribute_name: str = "Pearson") -> nx.Graph:
    """adds an attribute to each edge which is the Pearson correlation
    coefficient between the traces of the two pixels (nodes) associated
    with that edge.

    Parameters
    ----------
    graph: nx.Graph
        a graph with nodes like (row, col) and edges connecting them
    video_path: Path
        path to an hdf5 video file, assumed to have a dataset "data"
        nframes x nrow x ncol
    attribute_name: str
        name set on each edge for this calculated value

    Returns
    -------
    new_graph: nx.Graph
        an undirected networkx graph, with attribute added to edges

    """
    new_graph = nx.Graph()
    # copies over node attributes
    new_graph.add_nodes_from(graph.nodes(data=True))

    # load the section of data that encompasses this graph
    rows, cols = np.array(graph.nodes).T
    with h5py.File(video_path, "r") as f:
        data = f["data"][:,
                         rows.min(): (rows.max() + 1),
                         cols.min(): (cols.max() + 1)]

    offset = np.array([rows.min(), cols.min()])
    for node1 in graph:
        neighbors = set(list(graph.neighbors(node1)))
        new_neighbors = set(list(new_graph.neighbors(node1)))
        neighbors = list(neighbors - new_neighbors)
        if len(neighbors) == 0:
            continue
        nrow, ncol = np.array(node1) - offset
        irows, icols = (np.array(neighbors) - offset).T
        weights = 1.0 - cdist([data[:, nrow, ncol]],
                              [data[:, r, c] for r, c in zip(irows, icols)],
                              metric="correlation")[0]
        for node2, weight in zip(neighbors, weights):
            attr = graph.get_edge_data(node1, node2)
            attr.update({attribute_name: weight})
            new_graph.add_edge(node1, node2, **attr)

    return new_graph


def add_filtered_pearson_edge_attributes(
        filter_fraction: float,
        graph: nx.Graph,
        video_path: Path,
        attribute_name: str = "filtered_Pearson") -> nx.Graph:
    """adds an attribute to each edge which is the Pearson correlation
    coefficient between the traces of the two pixels (nodes) associated
    with that edge. Correlation coefficient is only calculated for the
    union of the brightest `filter_fraction` of timesteps at the two
    pixels

    Parameters
    ----------
    filter_fraction: float
        The fraction of timesteps to keep for each pixel when calculating
        the correlation coefficient
    graph: nx.Graph
        a graph with nodes like (row, col) and edges connecting them
    video_path: Path
        path to an hdf5 video file, assumed to have a dataset "data"
        nframes x nrow x ncol
    attribute_name: str
        name set on each edge for this calculated value

    Returns
    -------
    new_graph: nx.Graph
        an undirected networkx graph, with attribute added to edges

    """
    new_graph = nx.Graph()
    # copies over node attributes
    new_graph.add_nodes_from(graph.nodes(data=True))

    # load the section of data that encompasses this graph
    rows, cols = np.array(graph.nodes).T
    with h5py.File(video_path, "r") as f:
        data = f["data"][:,
                         rows.min(): (rows.max() + 1),
                         cols.min(): (cols.max() + 1)]

    offset = np.array([rows.min(), cols.min()])

    # create a lookup table of the brightest filter_fraction
    # timesteps for each pixel
    discard = 1.0 - filter_fraction
    i_threshold = np.round(discard * data.shape[0]).astype(int)

    for node1 in graph:
        n1row, n1col = np.array(node1) - offset
        flux1 = data[:, n1row, n1col]
        sorted_dex = np.argsort(flux1)
        mask1 = sorted_dex[i_threshold:]
        neighbors = set(list(graph.neighbors(node1)))
        new_neighbors = set(list(new_graph.neighbors(node1)))
        neighbors = list(neighbors - new_neighbors)
        if len(neighbors) == 0:
            continue

        for node2 in neighbors:
            n2row, n2col = np.array(node2) - offset
            flux2 = data[:, n2row, n2col]
            sorted_dex = np.argsort(flux2)
            mask2 = sorted_dex[i_threshold:]

            # create a global mask so that we are calculating the
            # correlation on the same timestamps for both pixels
            full_mask = np.unique(np.concatenate([mask1, mask2]))
            masked_flux1 = flux1[full_mask].astype(float)
            masked_flux2 = flux2[full_mask].astype(float)

            mu1 = np.mean(masked_flux1)
            mu2 = np.mean(masked_flux2)
            masked_flux1 -= mu1
            masked_flux2 -= mu2

            numerator = np.mean(masked_flux1 * masked_flux2)
            denom = np.mean(masked_flux1**2)
            denom *= np.mean(masked_flux2**2)

            attr = graph.get_edge_data(node1, node2)
            attr.update({attribute_name: numerator / np.sqrt(denom)})
            new_graph.add_edge(node1, node2, **attr)

    return new_graph


def neighborhood_pixels(coordinates: Tuple[float, float],
                        radius: float,
                        nrow: int,
                        ncol: int) -> np.ndarray:
    """return pixel coordinates of a circular region around a coordinate

    Parameters
    ----------
    coordinates: Tuple[float, float]
        the (row, col) coordinates that center the neighborhood
    radius: float
        the radius relative to 'coordinates' within which a pixel is
        considered in the neighborhood
    nrow: int
        the number of rows in the full FOV. Used to restrict return values
        to valid pixel coordinates in the FOV.
    ncol: int
        the number of columns in the full FOV. Used to restrict return values
        to valid pixel coordinates in the FOV.

    Returns
    -------
    coords: np.ndarray
        dtype np.int64 pixel coordinates within neighborhood of 'coordinates'

    """
    rmin = max(0, int(coordinates[0] - radius - 1))
    rmax = min(nrow - 1, int(coordinates[0] + radius + 1))
    cmin = max(0, int(coordinates[1] - radius - 1))
    cmax = min(ncol - 1, int(coordinates[1] + radius + 1))
    rows, cols = np.mgrid[rmin: (rmax + 1), cmin: (cmax + 1)]
    coords = np.array([np.array(rc) for rc in zip(rows.flat, cols.flat)])
    hypot = np.linalg.norm(coords - np.array(coordinates), axis=1)
    coords = coords[hypot <= radius]
    return coords


def _filtered_hnc_pearson(node0: np.ndarray,
                          node1: np.ndarray,
                          neighbor_traces: List[np.ndarray],
                          filter_fraction: float,
                          full_neighborhood: bool = False):
    """
    Correlate two pixels with their neighbors using only the brightest
    pixels. Note: a single mask representing the union of the brightest
    timesteps from all input traces will be used when calculating the
    correlation vectors for node0 and node1

    Parameters
    ----------
    node0: np.ndarray
        Trace of pixel 0

    node1: np.ndarray
        Trace of pixel 1

    neighbor_traces: List[np.ndarray]
        Each element is a trace from the neighborhood that node0 and node1
        need to be correlated with

    filter_fraction: float
        The fraction of pixels to keep

    full_neighborhood: bool
        If True, use all of the pixels in the neighborhood when selecting
        the timesteps to correlate. When False, only use node0 and node1
        (default = False)

    Returns
    -------
    pearsons: np.ndarray
        The result of running cdist on the input traces masked according
        to the specified filter_fraction
    """
    discard = 1.0 - filter_fraction
    thresh0 = np.quantile(node0, discard)
    mask0 = np.where(node0 >= thresh0)[0]
    thresh1 = np.quantile(node1, discard)
    mask1 = np.where(node1 >= thresh1)[0]

    global_mask = [mask0, mask1]
    if full_neighborhood:
        for trace in neighbor_traces:
            thresh = np.quantile(trace, discard)
            mask = np.where(trace >= thresh)[0]
            global_mask.append(mask)
    global_mask = np.unique(np.concatenate(global_mask))

    masked_node0 = node0[global_mask]
    masked_node1 = node1[global_mask]
    masked_neighbors = []
    for trace in neighbor_traces:
        masked_neighbors.append(trace[global_mask])

    pearsons = 1.0 - cdist([masked_node0, masked_node1],
                           masked_neighbors,
                           metric="correlation")

    return pearsons


def add_hnc_gaussian_metric(graph: nx.Graph,
                            video_path: Path,
                            neighborhood_radius: Union[int, float],
                            attribute_name: str = "hnc_Gaussian",
                            filter_fraction: Optional[float] = None,
                            full_neighborhood: bool = False) -> nx.Graph():
    """for each pair of pixels represented by an edge, calculates the Gaussian
    similarity in correlation space to a neighborhood of size
    'neighborhood_radius' where the correlation coefficient of all pairs of
    pixels in that neighborhood are considered.

    Parameters
    ----------
    graph: nx.Graph
        a graph with nodes like (row, col) and edges connecting them
    video_path: Path
        path to an hdf5 video file, assumed to have a dataset "data"
        nframes x nrow x ncol
    neighborhood_radius: Union[int, float]
        pixels within <= this radius of the pair centroid are considered
        in the neighborhood.
    attribute_name: str
        name set on each edge for this calculated value
    filter_fraction: Optional[float]
        If not None, take the brightest filter_fraction pixels when
        calculating the Pearson correlation within a neighborhood.
    full_neighborhood: bool
        If True, use all of the pixels in the neighborhood when selecting
        the timesteps to correlate. When False, only use node0 and node1
        (default = False)

    Returns
    -------
    new_graph: nx.Graph
        an undirected networkx graph, with attribute added to edges

    Notes
    -----
    See:
    "HNCcorr: A Novel Combinatorial Approach for Cell Identification
    in Calcium-Imaging Movies"
    Spaen et. al.
    https://pubmed.ncbi.nlm.nih.gov/31058211/

    """

    with h5py.File(video_path, "r") as f:
        nrow, ncol = f["data"].shape[1:]

    # find bounds of data to load
    row_min = np.inf
    row_max = 0
    col_min = np.inf
    col_max = 0
    for edge in graph.edges:
        neighborhood = neighborhood_pixels(np.array(edge).mean(axis=0),
                                           neighborhood_radius,
                                           nrow,
                                           ncol)
        n_mn = neighborhood.min(axis=0)
        n_mx = neighborhood.max(axis=0)
        if n_mn[0] < row_min:
            row_min = n_mn[0]
        if n_mn[1] < col_min:
            col_min = n_mn[1]
        if n_mx[0] > row_max:
            row_max = n_mx[0]
        if n_mx[1] > col_max:
            col_max = n_mx[1]

    # load the data
    with h5py.File(video_path, "r") as f:
        data = f["data"][:, row_min: (row_max + 1), col_min: (col_max + 1)]

    values = dict()
    for edge in graph.edges:
        neighborhood = neighborhood_pixels(np.array(edge).mean(axis=0),
                                           neighborhood_radius,
                                           nrow,
                                           ncol)
        traces = [data[:, row - row_min, col - col_min]
                  for row, col in neighborhood]
        node0 = data[:, edge[0][0] - row_min, edge[0][1] - col_min]
        node1 = data[:, edge[1][0] - row_min, edge[1][1] - col_min]

        if filter_fraction is None:
            pearsons = 1.0 - cdist([node0, node1],
                                   traces,
                                   metric="correlation")
        else:
            pearsons = _filtered_hnc_pearson(
                            node0,
                            node1,
                            traces,
                            filter_fraction,
                            full_neighborhood=full_neighborhood)

        dist = np.linalg.norm(pearsons[0] - pearsons[1])
        values[edge] = np.exp(-1.0 * (dist ** 2.0))
    nx.set_edge_attributes(G=graph, values=values, name=attribute_name)

    return graph
