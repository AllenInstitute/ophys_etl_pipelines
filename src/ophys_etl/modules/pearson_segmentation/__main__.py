import argschema
import networkx as nx
import numpy as np
import json
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import binary_dilation

from ophys_etl.modules.pearson_segmentation.schemas import \
        PearsonSegmentationInputSchema
from ophys_etl.qc.video.correlation_graph import CorrelationGraph
from ophys_etl.qc.video.utils import normalize_graph


def lims_dict_from_label_image(image):
    props = regionprops(image)
    rois = []
    for prop in props:
        roi = dict()
        roi["height"] = prop.bbox[2] - prop.bbox[0]
        roi["width"] = prop.bbox[3] - prop.bbox[1]
        roi["x"] = prop.bbox[1]
        roi["y"] = prop.bbox[0]
        mask = np.zeros((roi["height"], roi["width"])).astype(bool)
        mask[prop.coords[:, 0] - roi["y"],
             prop.coords[:, 1] - roi["x"]] = True
        roi["mask_matrix"] = [i.tolist() for i in mask]
        rois.append(roi)
    return rois


def merge_rois(image):
    props = regionprops(image)
    merge_list = []
    for i in range(len(props)):
        ima = np.zeros_like(image)
        ima[props[i].coords[:, 0], props[i].coords[:, 1]] = 1
        ima = binary_dilation(ima)
        labels = np.unique(image[ima != 0])
        labels = set(np.delete(labels, 0))
        added = False
        for m in merge_list:
            if labels & m:
                m.update(labels)
                added = True
        if not added:
            merge_list.append(labels)
    for m in merge_list:
        label = m.pop()
        for i in m:
            image[image == i] = label
    return image


def roi_filter(lims_dict):
    for roi in lims_dict:
        npix = np.count_nonzero(roi["mask_matrix"])
        valid = False
        if (npix > 25) & (npix < 500):
            valid = True
        if (roi["width"] < 3) | (roi["height"] < 3):
            valid = False
        roi["valid_roi"] = valid
    return lims_dict


class PearsonSegmentation(argschema.ArgSchemaParser):
    default_schema = PearsonSegmentationInputSchema

    def run(self):
        self.logger.name = type(self).__name__

        # create graph if not provided
        if "graph_input" not in self.args:
            cg = CorrelationGraph(input_data=self.args["graph_creation"],
                                  args=[])
            cg.run()
            self.args["graph_input"] = \
                self.args["graph_creation"]["graph_output"]

        # read and optionally normalize graph
        graph = nx.read_gpickle(self.args["graph_input"])
        if self.args["normalize"]:
            graph = normalize_graph(graph, self.args["sigma"])

        # determine normalized variance
        coords = np.array(list(graph.nodes))
        shape = tuple(coords.max(axis=0) + 1)
        stdev_over_mean = np.zeros(shape)
        for node in graph.nodes:
            vals = [graph[node][i]["weight"] for i in graph.neighbors(node)]
            stdev_over_mean[node[0], node[1]] = np.std(vals) / np.mean(vals)
        threshold = np.quantile(stdev_over_mean.flat,
                                self.args["mask_quantile"])
        mask = np.ones(shape).astype(int)
        mask[stdev_over_mean > threshold] = 0
        seg = watershed(stdev_over_mean, mask=mask)
        seg = merge_rois(seg)

        # plt.imshow(seg)
        # plt.show()

        rois = lims_dict_from_label_image(seg)
        rois = roi_filter(rois)

        with open(self.args["roi_output"], "w") as f:
            json.dump(rois, f, indent=2)

        self.logger.info(f"wrote {self.args['roi_output']}")


if __name__ == "__main__":  # pragma: nocover
    gs = PearsonSegmentation()
    gs.run()
