import argschema
import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sknetwork.clustering import Louvain

from ophys_etl.modules.graph_segmentation.schemas import \
        GraphSegmentationInputSchema


class GraphSegmentation(argschema.ArgSchemaParser):
    default_schema = GraphSegmentationInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        graph = nx.read_gpickle(self.args["graph_input"])
        csr = nx.to_scipy_sparse_matrix(graph)
        louvain = Louvain(
                modularity='newman',
                resolution=self.args["louvain_resolution"],
                shuffle_nodes=True,
                #tol_optimization=1e-9,
                #tol_aggregation=1e-9,
                verbose=True)
        labels = louvain.fit_transform(csr)
        z = np.zeros((512, 512), dtype='float')
        mz = np.zeros((512, 512), dtype='int')
        nodes = np.array(list(graph.nodes))
        ulabels, counts = np.unique(labels, return_counts=True)
        for label, count in list(zip(ulabels, counts)):
            inds = np.argwhere(labels == label)
            lnodes = nodes[inds].squeeze()
            #sub_graph = graph.subgraph(nodes[inds])
            sub_graph = graph.subgraph([n for i, n in enumerate(graph.nodes) if i in inds])
            v = sum([v["weight"] for v in sub_graph.edges.values()])
            v /= len(sub_graph.edges)
            z[lnodes[:, 0], lnodes[:, 1]] = v
            mz[lnodes[:, 0], lnodes[:, 1]] = label

        levels = np.concatenate([ulabels, [ulabels[-1] + 1]]) - 0.5

        cmap, norm = matplotlib.colors.from_levels_and_colors(
                levels=levels,
                colors=[matplotlib.cm.tab20(i % 20) for i in ulabels])

        f, a = plt.subplots(1, 2, clear=True, num=1, sharex=True, sharey=True)
        a[0].imshow(mz, cmap=cmap, norm=norm)
        vmin = z[z != 0].min()
        vmax = z[z != 0].max()
        a[1].imshow(z, cmap="plasma", vmin=vmin, vmax=vmax)
        f.suptitle(f"louvain resolution {self.args['louvain_resolution']}")
        plt.show()


if __name__ == "__main__":  # pragma: nocover
    gs = GraphSegmentation()
    gs.run()
