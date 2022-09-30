import numpy as np
import tifffile
import pathlib
import tempfile


def _create_full_field_tiff(
        numVolumes: int,
        numSlices: int,
        seed: int,
        output_dir: pathlib.Path,
        nrows: int = 23,
        ncols: int = 37):
    """
    Create a random full field tiff according to specified
    metadata fields

    Parameters
    ----------
    numVolmes: int
        value of SI.hStackManager.actualNumVolumes

    numSlices: int
        value of SI.hStackmanager.actualNumSlices

    seed: int
        Seed passed to random number generator

    output_dir: pathlib.Path
        directory where output files can be written

    nrows: int
        Number of rows in the image averaged over pages

    ncols: int
        Number of columns in the image averaged over pages

    Returns
    -------
    tiff_path: pathlib.Path
        path to example full field TIFF

    avg_img: np.ndarray
        expected array resulting from taking the average over
        slices and volumes

    metadata:
        List containing dict of metadata fields
    """
    rng = np.random.default_rng(seed=seed)
    data = rng.random((numVolumes*numSlices, nrows, ncols))
    avg_img = data.mean(axis=0)
    tiff_pages = [data[ii, :, :] for ii in range(data.shape[0])]
    tiff_path = pathlib.Path(
            tempfile.mkstemp(dir=output_dir,
                             suffix='.tiff')[1])
    tifffile.imsave(tiff_path, tiff_pages)
    metadata = [{'SI.hStackManager.actualNumVolumes': numVolumes,
                 'SI.hStackManager.actualNumSlices': numSlices}]

    return (tiff_path, avg_img, metadata)
