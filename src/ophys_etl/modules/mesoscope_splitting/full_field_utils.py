import pathlib
import numpy as np
import tifffile
from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)


def _average_full_field_tiff(
        tiff_path: pathlib.Path) -> np.ndarray:
    """
    Read in the image data from a fullfield TIFF image and average
    over slices and volumes.

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the TIFF file

    Returns
    -------
    avg_img: np.ndarray
    """
    metadata = ScanImageMetadata(tiff_path)

    with tifffile.TiffFile(tiff_path, mode='rb') as in_file:
        data = np.stack([p.asarray() for p in in_file.pages])

    # validate that the number of pages in the tiff file
    # was as expected
    expected_n_pages = metadata.numVolumes*metadata.numSlices
    if data.shape[0] != expected_n_pages:
        msg = f"{tiff_path}\n"
        msg += f"numVolumes: {metadata.numVolumes}\n"
        msg += f"numSlices: {metadata.numSlices}\n"
        msg += f"implies n_pages: {expected_n_pages}\n"
        msg += f"actual n_pages: {data.shape[0]}"
        raise ValueError(msg)

    return data.mean(axis=0)
