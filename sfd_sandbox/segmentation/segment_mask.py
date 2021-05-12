import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import PIL.Image
import numpy as np
import json
import networkx as nx
import argparse
import time
from equalizer import AdaptiveEqualizer

from numpy.fft import fft2, ifft2

from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI

from ophys_etl.modules.segmentation.qc_utils.roi_utils import (
    add_roi_boundaries_to_img,
    add_labels_to_roi_img)

class Painter(object):

    def __init__(self, mask, start_pt):
        self._mask = mask
        self._row_max = self._mask.shape[0]
        self._col_max = self._mask.shape[1]
        self._to_paint_by_rows = set()
        self._to_paint_by_cols = set()
        self._to_paint_by_NE = set()
        self._to_paint_by_NW = set()
        self._painted_by_rows = set()
        self._painted_by_cols = set()
        self._painted_by_NE = set()
        self._painted_by_NW = set()
        self._roi_pixels = set()
        self._roi_pixels.add(start_pt)
        self._global_start = start_pt
        self._scans = 0
        self._aborted = 0
        self._void = 0

    def _pt_valid(self, pt):
        if pt[0] < 0:
            return False
        if pt[0] >= self._row_max:
            return False
        if pt[1] < 0:
            return False
        if pt[1] >= self._col_max:
            return False
        return True

    def _log_void(self, pt):
        pt = tuple(pt)
        for _arch in (self._painted_by_rows,
                      self._painted_by_cols,
                      self._painted_by_NW,
                      self._painted_by_NE):
            _arch.add(pt)

    def _paint(self, start_pt, direction):

        if not self._mask[start_pt[0], start_pt[1]]:
            self._log_void(start_pt)
            self._void += 1
            return None

        if direction == 'row':
            delta = (1, 0)
            archive = self._painted_by_rows
        elif direction == 'col':
            delta = (0, 1)
            archive = self._painted_by_cols
        elif direction == 'nw':
            delta = (1, 1)
            archive = self._painted_by_NW
        elif direction == 'ne':
            delta = (1, -1)
            archive = self._painted_by_NE
        else:
            raise RuntimeError(f"cannot parse direction {direction}")

        if start_pt in archive:
            self._aborted += 1
            return None

        px = tuple(start_pt)
        self._roi_pixels.add(px)
        archive.add(px)

        for norm in (1, -1):
            pt = [start_pt[0], start_pt[1]]
            pt[0] += norm*delta[0]
            pt[1] += norm*delta[1]
            while self._pt_valid(pt):
                self._scans += 1
                if not self._mask[pt[0], pt[1]]:
                    self._void += 1
                    self._log_void(pt)
                    break
                px = tuple(pt)
                self._roi_pixels.add(px)
                archive.add(px)
                if direction != 'row' and px not in self._painted_by_rows:
                    self._to_paint_by_rows.add(px)
                if direction != 'col' and px not in self._painted_by_cols:
                    self._to_paint_by_cols.add(px)
                if direction != 'nw' and px not in self._painted_by_NW:
                    self._to_paint_by_NW.add(px)
                if direction != 'ne' and px not in self._painted_by_NE:
                    self._to_paint_by_NE.add(px)
                pt[0] += norm*delta[0]
                pt[1] += norm*delta[1]
        return None

    def tot_targets(self):
        n = len(self._to_paint_by_rows)
        n += len(self._to_paint_by_cols)
        n += len(self._to_paint_by_NW)
        n += len(self._to_paint_by_NE)
        return n

    def jsonize(self):
        n_pix = len(self._roi_pixels)
        row = np.zeros(n_pix, dtype=int)
        col = np.zeros(n_pix, dtype=int)
        for ii, px in enumerate(self._roi_pixels):
            row[ii] = px[0]
            col[ii] = px[1]
        row_min = row.min()
        row_max = row.max()
        col_min = col.min()
        col_max = col.max()
        mask = np.zeros((row_max-row_min+1,
                         col_max-col_min+1), dtype=bool)
        mask[row-row_min, col-col_min] = True

        roi ={'x0': col_min,
              'y0': row_min,
              'mask_matrix': list([list(mask[ii,:])
                                   for ii in range(mask.shape[0])]),
              'height': mask.shape[0],
              'width': mask.shape[1]}

        return {'roi': roi,
                'row': row,
                'col': col}

    def run(self):
        start_pt = self._global_start
        for d in ('row', 'col', 'nw', 'ne'):
            self._paint(start_pt, d)

        while self.tot_targets() > 0:
            if len(self._to_paint_by_rows) > 0:
                pt = self._to_paint_by_rows.pop()
                self._paint(pt, 'row')
            if len(self._to_paint_by_cols) > 0:
                pt = self._to_paint_by_cols.pop()
                self._paint(pt, 'col')
            if len(self._to_paint_by_NW) > 0:
                pt = self._to_paint_by_NW.pop()
                self._paint(pt, 'nw')
            if len(self._to_paint_by_NE) > 0:
                pt = self._to_paint_by_NE.pop()
                self._paint(pt, 'ne')

        return self.jsonize()


class MaskSegmenter(object):

    def __init__(self, mask):
        """
        Parameters
        ----------
        mask: np.ndarray
            A mask of booleans representing the image to be segmented
        """
        self._mask = np.copy(mask)
        self._rois = []

    def run(self):
        while self._mask.sum() > 0:
            valid = np.where(self._mask)
            start_pt = (valid[0][0], valid[1][0])
            the_painter = Painter(self._mask, start_pt)
            roi = the_painter.run()
            self._mask[roi['row'], roi['col']] = False
            self._rois.append(roi['roi'])

        return self._rois


def graph_to_img(graph_fname):
    graph = nx.read_gpickle(graph_fname)
    coords = np.array(list(graph.nodes))
    shape = tuple(coords.max(axis=0) + 1)
    img = np.zeros(shape)
    for node in graph.nodes:
        vals = [graph[node][i]["weight"] for i in graph.neighbors(node)]
        img[node[0], node[1]] = np.sum(vals)

    return img


def mask_img(img, frac):
    new_img = np.ones(img.shape, dtype=bool)
    flux_arr = np.sort(img.flatten())
    ii = np.round(frac*len(flux_arr)).astype(int)
    mask = img<flux_arr[ii]
    new_img[mask] = False
    return new_img


def img_to_rgb(img):
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=int)
    vmax = img.max()
    vmin = img.min()
    for ic in range(3):
        rgb_img[:, :, ic] = np.round(255*((img-vmin)/(vmax-vmin))).astype(int)
    rgb_img = np.where(rgb_img<=255, rgb_img, 255)
    #print(vmin,vmax)
    #print(rgb_img.min())
    #print(np.median(rgb_img))
    #print(rgb_img.max())
    return rgb_img

def read_old_roi(file_path):
    roi_list = []
    with open(file_path, 'rb') as in_file:
        json_data = json.load(in_file)
    for roi_id, data in enumerate(json_data['rois']):
        if not data['valid']:
            continue
        roi = OphysROI(x0=data['x'],
                       y0=data['y'],
                       height=data['height'],
                       width=data['width'],
                       mask_matrix=data['mask'],
                       valid_roi=True,
                       roi_id=-1-roi_id)
        roi_list.append(roi)
    return roi_list


def isolate_low_frequency_modes(img, n_modes):
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

    mask = np.where(freq_grid>freq_arr[n_modes])
    transformed[mask] = 0.0
    new_img = ifft2(transformed)
    return new_img.real


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_graph', type=str, default=None)
    parser.add_argument('--out_img', type=str, default=None)
    parser.add_argument('--max_proj', type=str, default=None)
    parser.add_argument('--old_roi', type=str, default=None)
    args = parser.parse_args()

    graph_img = graph_to_img(args.in_graph)
    equalizer = AdaptiveEqualizer(graph_img, (10, 10))

    subtracted_img = equalizer.equalize(graph_img, masking=True)
    #bckgd = isolate_low_frequency_modes(graph_img, 10)
    #subtracted_img = graph_img-bckgd

    if args.max_proj is None:
        rgb_img = img_to_rgb(subtracted_img)
    else:
        rgb_img = PIL.Image.open(args.max_proj, 'r')
        nr = rgb_img.size[0]
        nc = rgb_img.size[1]
        rgb_img = np.array(rgb_img).reshape(nr,nc)
        rgb_img = img_to_rgb(rgb_img)

    graph_mask = mask_img(subtracted_img, 0.9)
    #graph_mask = equalizer.mask(graph_img, 0.95)

    fig, ax = plt.subplots(1, 1, figsize=(20,20))
    ax.imshow(graph_mask)
    fig.savefig('mask.png')
    plt.close(fig=fig)

    t0 = time.time()
    segmenter = MaskSegmenter(graph_mask)
    base_roi_list = segmenter.run()
    print('segmentation took %e seconds' % (time.time()-t0))

    roi_list = []
    for roi_id, roi_data in enumerate(base_roi_list):
        roi = OphysROI(x0=int(roi_data['x0']),
                       y0=int(roi_data['y0']),
                       height=int(roi_data['height']),
                       width=int(roi_data['width']),
                       mask_matrix=roi_data['mask_matrix'],
                       roi_id=roi_id,
                       valid_roi=True)
        roi_list.append(roi)

    rgb_img = add_roi_boundaries_to_img(rgb_img, roi_list, alpha=0.5)
    if args.old_roi is not None:
        old_roi_list = read_old_roi(args.old_roi)
        rgb_img = add_roi_boundaries_to_img(rgb_img,
                                            old_roi_list,
                                            color=(0,255,0),
                                            alpha=0.5)

    fig, ax = plt.subplots(1, 1, figsize=(20,20))
    ax.imshow(rgb_img)
    ax = add_labels_to_roi_img(ax,
                               [{'color': (255, 0, 0),
                                 'rois': roi_list}])
    fig.savefig(args.out_img)
    plt.close(fig=fig)
