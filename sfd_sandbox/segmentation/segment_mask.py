import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import json


class Painter(object):

    def __init__(self, mask, start_pt):
        self._mask = mask
        self._row_max = self._mask.shape[0]
        self._col_max = self._mask.shape[1]
        self._to_paint_by_rows = []
        self._to_paint_by_cols = []
        self._to_paint_by_NE = []
        self._to_paint_by_NW = []
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

    def _paint(self, start_pt, direction):
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
        if not self._mask[start_pt[0], start_pt[1]]:
            self._void += 1
            return None

        for norm in (1, -1):
            pt = [start_pt[0], start_pt[1]]
            pt[0] += norm*delta[0]
            pt[1] += norm*delta[1]
            while self._pt_valid(pt):
                self._scans += 1
                if not self._mask[pt[0], pt[1]]:
                    self._void += 1
                    break
                px = tuple(pt)
                self._roi_pixels.add(px)
                archive.add(px)
                if direction != 'row' and px not in self._painted_by_rows:
                    self._to_paint_by_rows.append(px)
                if direction != 'col' and px not in self._painted_by_cols:
                    self._to_paint_by_cols.append(px)
                if direction != 'nw' and px not in self._painted_by_NW:
                    self._to_paint_by_NW.append(px)
                if direction != 'ne' and px not in self._painted_by_NE:
                    self._to_paint_by_NE.append(px)
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
        return {'x0': col_min,
                'y0': row_min,
                'mask_matrix': list([list(mask[ii,:])
                                     for ii in range(mask.shape[0])]),
                'height': mask.shape[0],
                'width': mask.shape[1]}

    def run(self):
        start_pt = self._global_start
        for d in ('row', 'col', 'nw', 'ne'):
            self._paint(start_pt, d)

        while self.tot_targets() > 0:
            if len(self._to_paint_by_rows) > 0:
                pt = self._to_paint_by_rows.pop(0)
                self._paint(pt, 'row')
            if len(self._to_paint_by_cols) > 0:
                pt = self._to_paint_by_cols.pop(0)
                self._paint(pt, 'col')
            if len(self._to_paint_by_NW) > 0:
                pt = self._to_paint_by_NW.pop(0)
                self._paint(pt, 'nw')
            if len(self._to_paint_by_NE) > 0:
                pt = self._to_paint_by_NE.pop(0)
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
        self._base_mask = np.copy(mask)
        self._rois = []


if __name__ == "__main__":

    mask = np.zeros((20, 20), dtype=bool)

    mask[4:8, 11:15] = True
    mask[5, 14] = True
    mask[5,15] = True
    mask[5,16] = True
    mask[5,17] = True
    mask[6, 17] = True
    mask[7, 17] = True
    mask[8, 17] = True
    mask[7, 14] = True
    mask[8, 14] = True
    mask[9, 14] = True
    mask[10, 14] = True
    mask[11, 14] = True
    mask[11, 15] = True
    mask[11, 16] = True
    mask[11, 17] = True
    mask[6, 12] = False
    mask[3, 10] = True
    mask[2, 9] = True
    mask[1, 8] = True
    mask[1, 7] = True
    mask[1, 6] = True
    mask[2, 6] = True
    mask[3, 6] = True
    mask[1, 5] = True
    mask[1, 6] = True
    mask[1, 7] = True
    mask[2, 4] = True
    mask[1, 3] = True

    f, a = plt.subplots(1,1,figsize=(10,10))
    a.imshow(mask)
    f.savefig('junk.png')

    p = Painter(mask, (6, 13))
    roi = p.run()
    print('scans ',p._scans,mask[6,13])
    print('aborted ',p._aborted)
    print('void ',p._void)
    print(mask.sum())
    print(len(p._roi_pixels))
    assert len(p._roi_pixels) == mask.sum()
    for p in p._roi_pixels:
        assert mask[p[0], p[1]]

    new_mask = roi['mask_matrix']
    r0 = roi['y0']
    c0 = roi['x0']
    for irow in range(len(new_mask)):
        r = r0+irow
        for icol in range(len(new_mask[0])):
            c = c0 + icol
            if new_mask[irow][icol]:
                assert mask[r, c]
            else:
                assert not mask[r, c]

