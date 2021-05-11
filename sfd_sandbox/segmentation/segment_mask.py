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

    def paint_by_row(self, start_pt):
        self._scans += 1
        if start_pt in self._painted_by_rows:
            return None
        if not self._mask[start_pt[0], start_pt[1]]:
            return None

        icol = start_pt[1]
        for drow in (-1, 1):
            irow = start_pt[0] + drow
            while irow >= 0 and irow < self._row_max:
                if not self._mask[irow, icol]:
                    break
                px = (irow, icol)
                self._roi_pixels.add(px)
                self._scans += 1
                if px not in self._painted_by_cols:
                    self._to_paint_by_cols.append(px)
                self._painted_by_rows.add(px)
                irow += drow
        return None

    def paint_by_col(self, start_pt):
        self._scans += 1
        if start_pt in self._painted_by_cols:
            return None
        if not self._mask[start_pt[0], start_pt[1]]:
            return None

        irow = start_pt[0]
        for dcol in (-1, 1):
            icol = start_pt[1] + dcol
            while icol >= 0 and icol < self._col_max:
                if not self._mask[irow, icol]:
                    break
                px = (irow, icol)
                self._roi_pixels.add(px)
                self._scans += 1
                if px not in self._painted_by_rows:
                    self._to_paint_by_rows.append(px)
                self._painted_by_cols.add(px)
                icol += dcol
        return None

    def tot_targets(self):
        n = len(self._to_paint_by_rows)
        n += len(self._to_paint_by_cols)
        n += len(self._to_paint_by_NW)
        n += len(self._to_paint_by_NE)
        return n

    def run(self):
        start_pt = self._global_start
        self.paint_by_row(start_pt)
        self.paint_by_col(start_pt)

        ct = 0

        while self.tot_targets() > 0:

            if len(self._to_paint_by_rows) > len(self._to_paint_by_cols):
                start_pt = self._to_paint_by_rows.pop(0)
                self.paint_by_row(start_pt)
            else:
                start_pt = self._to_paint_by_cols.pop(0)
                self.paint_by_col(start_pt)


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

    f, a = plt.subplots(1,1,figsize=(10,10))
    a.imshow(mask)
    f.savefig('junk.png')

    p = Painter(mask, (6, 13))
    p.run()
    print('scans ',p._scans,mask[6,13])
    assert len(p._roi_pixels) == mask.sum()
    for p in p._roi_pixels:
        assert mask[p[0], p[1]]

