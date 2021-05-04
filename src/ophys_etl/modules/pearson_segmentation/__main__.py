import argschema
import networkx as nx
import numpy as np
from numpy.fft import fft2, ifft2, fftfreq
import json
from skimage.segmentation import watershed
from skimage.measure import regionprops
from skimage.morphology import binary_dilation

from ophys_etl.modules.pearson_segmentation.schemas import \
        PearsonSegmentationInputSchema
from ophys_etl.qc.video.correlation_graph import CorrelationGraph
from ophys_etl.qc.video.utils import normalize_graph


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def fit_background(img):
    transformed = fft2(img)

    ty = transformed.shape[0]
    tx = transformed.shape[1]

    fx = np.zeros(tx, dtype=int)
    fy = np.zeros(ty, dtype=int)
    for ii in range(tx):
        if ii<tx//2:
            fx[ii] = ii
        else:
            fx[ii] = tx-1-ii
    for ii in range(ty):
        if ii<ty//2:
            fy[ii] = ii
        else:
            fy[ii] = ty-1-ii

    freq_grid = np.meshgrid(fx, fy)

    freq_grid = freq_grid[0]**2+freq_grid[1]**2

    assert freq_grid.shape == transformed.shape

    freq_arr = np.unique(freq_grid.flatten())
    nf = len(freq_arr)

    f0 = freq_arr[5]
    #f1 = freq_arr[nf-200]
    #print('\nfreq')
    #print(f0,f1)
    #print(nf)

    #mask = np.logical_and(freq_grid>f0,
    #                      freq_grid<f1)

    mask = freq_grid>f0

    #d = 45
    #ty0 = ty//2-22*ty//d
    #ty1 = ty//2+22*ty//d
    #tx0 = tx//2-22*tx//d
    #tx1 = tx//2+22*tx//d

    #transformed[ty0:ty1, :] = 0.0
    #transformed[:, tx0:tx1] = 0.0
    #transformed[ty//2, tx//2] = 0.0

    transformed[mask] = 0.0

    bckgd = ifft2(transformed).real

    diff = img-bckgd

    m = np.mean(img)
    print('img ',((img-m)**2).sum(),img.min(),img.max(),np.median(img))
    print('subtr ',(diff**2).sum(),diff.min(),diff.max(),np.median(diff))
    return bckgd


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
        #coords = np.array(list(graph.nodes))
        #shape = tuple(coords.max(axis=0) + 1)
        #stdev_over_mean = np.zeros(shape)
        #for node in graph.nodes:
        #    vals = [graph[node][i]["weight"] for i in graph.neighbors(node)]
        #    stdev_over_mean[node[0], node[1]] = np.std(vals) / np.mean(vals)
        #threshold = np.quantile(stdev_over_mean.flat,
        #                         self.args["mask_quantile"])
        #mask = np.ones(shape).astype(int)
        #mask[stdev_over_mean > threshold] = 0
        #seg = watershed(stdev_over_mean, mask=mask)

        coords = np.array(list(graph.nodes))
        shape = tuple(coords.max(axis=0) + 1)
        img = np.zeros(shape)
        for node in graph.nodes:
            vals = [graph[node][i]["weight"] for i in graph.neighbors(node)]
            img[node[0], node[1]] = np.median(vals)

        # build mask
        dx = 200
        dy = 200
        fig_ct = 0
        mask = np.zeros(shape).astype(int)
        for xx in range(dx//2, shape[0], dx//2):
            xmin = xx-dx//2
            xmax = xx+dx//2
            for yy in range(dy//2, shape[1], dy//2):
                ymin = yy-dy//2
                ymax = yy+dy//2

                window = img[xmin:xmax, ymin:ymax]
                #print(window.shape)

                f, a = plt.subplots(3,1,figsize=(10,10))
                a = a.flatten()

                a[0].imshow(window)
                a[0].set_title("img", fontsize=10)

                bckgd = fit_background(window)

                subtracted_window = (window-bckgd).flatten()
                n = window.shape[0]*window.shape[1]
                mu = np.median(subtracted_window)
                std = np.sqrt(np.sum((subtracted_window-mu)**2)/(n-1))
                valid = np.where(window-bckgd>mu+std)

                a[1].imshow(bckgd)
                a[1].set_title("background",fontsize=10)

                a[2].imshow((window-bckgd-mu)/std)
                a[2].set_title('diff', fontsize=10)

                for axis in a:
                    axis.tick_params(which='both', axis='both',
                             left=0, bottom=0,
                             labelleft=0, labelbottom=0)
                    for s in ('top', 'bottom', 'left', 'right'):
                        axis.spines[s].set_visible(False)
                f.tight_layout()
                f.savefig(f'window_figs/window_{xmin}_{ymin}.png')
                plt.close(fig=f)
                #if fig_ct>10:
                #    exit()
                fig_ct += 1

                print('outside ',np.sum(subtracted_window**2))

                mask[xmin:xmax, ymin:ymax][valid] = 1

        seg = watershed(img, mask=mask)
        print('seg')
        print(seg)
        print(mask)
        print(mask.sum())
        print(seg.shape)
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
