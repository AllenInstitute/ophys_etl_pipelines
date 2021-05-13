import numpy as np


def add_gridlines(img_array: np.ndarray,
                  denom: int) -> np.ndarray:
    """
    Add grid lines to an image

    Parameters
    ----------
    img_array: np.ndarray
        The input image

    denom: int
        1/alpha for the grid lines

    Returns
    -------
    out_array: np.ndarray
        The image with gridline superimposed
    """
    out_array = np.copy(img_array)
    nrows = img_array.shape[0]
    ncols = img_array.shape[1]
    for ix in range(nrows//4, nrows-4, nrows//4):
        for ic in range(3):
            v = out_array[ix, :, ic]
            new = ((denom-1)*v+255)//denom
            out_array[ix, :, ic] = new

    for iy in range(ncols//4, ncols-4, ncols//4):
        for ic in range(3):
            v = out_array[:, iy, ic]
            new = ((denom-1)*v+255)//denom
            out_array[:, iy, ic] = new

    return out_array
