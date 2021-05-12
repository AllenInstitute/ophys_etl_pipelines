import numpy as np


def make_cdf(img_flat):
    val, val_ct = np.unique(img_flat, return_counts=True)
    cdf = np.cumsum(val_ct)
    cdf = cdf/val_ct.sum()
    assert len(val) == len(cdf)
    assert cdf.max()<=1.0
    assert cdf.min()>=0.0
    return val, cdf


def equalize_img(img):
    f = img.flatten()
    val, cdf = make_cdf(f)
    new_img = np.interp(f, val, cdf)
    return new_img.reshape(img.shape)


class EqTile(object):

    def __init__(self,
                 origin,
                 row_bounds,
                 col_bounds,
                 sub_img):
        """
        bounds of form [min, max)
        """
        assert sub_img.shape == (row_bounds[1]-row_bounds[0],
                                 col_bounds[1]-col_bounds[0])

        (self.val,
         self.cdf) = make_cdf(sub_img.flatten())
        self.origin = origin
        self.row_bounds = row_bounds
        self.col_bounds = col_bounds

    def interp(self, img):
        f = img.flatten()
        eq_img = np.interp(f, self.val, self.cdf,
                           left=1, right=0)
        return eq_img.reshape(img.shape)

    def center_dist(self, img):
        (rows,
         cols) = np.meshgrid(np.arange(img.shape[0]),
                             np.arange(img.shape[1]),
                             indexing='ij')

        return np.sqrt((rows-self.origin[0])**2
                       + (cols-self.origin[1])**2)

    def is_in(self, img):
        (rows,
         cols) = np.meshgrid(np.arange(img.shape[0]),
                             np.arange(img.shape[1]),
                             indexing='ij')

        in_row = np.logical_and(rows >= self.row_bounds[0],
                                rows < self.row_bounds[1])
        in_col = np.logical_and(cols >= self.col_bounds[0],
                                cols < self.col_bounds[1])
        return np.logical_and(in_row, in_col)


class AdaptiveEqualizer(object):

    def __init__(self, img, tile_grid):
        tile_params = self.get_tile_params(img.shape, tile_grid)
        self.tiles = []
        for p in tile_params:
            sub_img = img[p['row_bounds'][0]:p['row_bounds'][1],
                          p['col_bounds'][0]:p['col_bounds'][1]]
            t = EqTile(p['origin'],
                       row_bounds=p['row_bounds'],
                       col_bounds=p['col_bounds'],
                       sub_img=sub_img)
            self.tiles.append(t)

    def get_tile_params(self, img_shape, tile_grid):
        drow = img_shape[0]//tile_grid[0]
        dcol = img_shape[1]//tile_grid[1]

        tile_params = []
        row_vals = list(range(drow//4, img_shape[0], drow))
        col_vals = list(range(dcol//4, img_shape[1], dcol))
        for i_row, row in enumerate(row_vals):
            row_min = max(0, row - drow//2)
            row_max = min(img_shape[0], row_min+3*drow//2)
            if i_row == len(row_vals):
                row_max = img_shape[0]
            for i_col, col in enumerate(col_vals):
                col_min = max(0, col - dcol//2)
                col_max = min(img_shape[1], col + 3*dcol//2)
                if i_col == len(col_vals):
                    col_max = img._shape[1]
                obj = {'origin': (row, col),
                       'row_bounds': (row_min, row_max),
                       'col_bounds': (col_min, col_max)}
                tile_params.append(obj)
        return tile_params


    def equalize(self, img, masking=False):
        wgt = np.zeros(img.shape, dtype=float)
        new_img = np.zeros(img.shape, dtype=float)
        mask = np.ones(img.shape, dtype=bool)
        for t in self.tiles:
            w = 1.0/((1.0+t.center_dist(img))**2)
            if masking:
                mask = t.is_in(img)
            assert w.shape == img.shape
            wgt[mask] += w[mask]
            new_img[mask] += w[mask]*t.interp(img)[mask]
        return new_img/wgt

    def mask(self, img, frac=0.95):
        mask_img = np.zeros(img.shape, dtype=bool)
        for t in self.tiles:
            is_in = t.is_in(img)
            sub_img = t.interp(img[is_in])
            f = sub_img.flatten()
            f = np.sort(f)
            threshold = f[np.round(frac*len(f)).astype(int)]
            sub_mask = (sub_img>=threshold)
            mask_img[is_in] = np.logical_or(mask_img[is_in],
                                            sub_mask)
        return mask_img
