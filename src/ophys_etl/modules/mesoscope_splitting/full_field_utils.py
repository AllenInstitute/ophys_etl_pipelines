import pathlib
from typing import Dict
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

    # accumulate the pages one-by-one to avoid loading a
    # large numpy array into memory needlessly
    page_ct = 0
    avg_img = None
    with tifffile.TiffFile(tiff_path, mode='rb') as in_file:
        for page in in_file.pages:
            page_ct += 1

            # must cast as float to avoid overflow errors
            arr = page.asarray().astype(float)

            if avg_img is None:
                avg_img = arr
            else:
                avg_img += arr

    avg_img = avg_img / page_ct

    # validate that the number of pages in the tiff file
    # was as expected
    expected_n_pages = metadata.numVolumes*metadata.numSlices
    if page_ct != expected_n_pages:
        msg = f"{tiff_path}\n"
        msg += f"numVolumes: {metadata.numVolumes}\n"
        msg += f"numSlices: {metadata.numSlices}\n"
        msg += f"implies n_pages: {expected_n_pages}\n"
        msg += f"actual n_pages: {page_ct}"
        raise ValueError(msg)

    return avg_img


def _get_stitched_tiff_shapes(
        tiff_path: pathlib.Path,
        avg_img: np.ndarray) -> Dict:
    """
    Get the final shape for the stitched TIFF to be produced
    for a given full field TIFF

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the full field TIFF

    avg_img: np.ndarray
        Average image produced by _average_full_field_tiff

    Returns
    -------
    shape_dict: Dict
        'shape': the final shape of the stitched tiff
        'gap': the gap (in pixels) between columns in the final stitched image
    """

    metadata = ScanImageMetadata(tiff_path)

    # Make sure that every ROI only has one scanfield
    for roi in metadata.defined_rois:
        if not isinstance(roi['scanfields'], dict):
            msg = f"{tiff_path}\n"
            msg += "contains an ROI with more than one scanfield;\n"
            msg += "uncertain how to handle this case"
            raise ValueError(msg)

    # Make sure that every ROI has the same size in pixels as determined
    # by pixelResolutionXY
    baseline_pixel_x = None
    baseline_pixel_y = None
    for roi in metadata.defined_rois:
        this_x = roi['scanfields']['pixelResolutionXY'][0]
        this_y = roi['scanfields']['pixelResolutionXY'][1]
        if baseline_pixel_x is None:
            baseline_pixel_x = this_x
        if baseline_pixel_y is None:
            baseline_pixel_y = this_y
        if baseline_pixel_x != this_x or baseline_pixel_y != this_y:
            msg = f"{tiff_path}\n"
            msg += "contains ROIs with different pixel resolutions;\n"
            msg += "uncertain how to handle this case"
            raise ValueError(msg)

    n_rois = len(metadata.defined_rois)

    stitched_shape = (baseline_pixel_x*n_rois,
                      baseline_pixel_y)

    gap = (avg_img.shape[0] - baseline_pixel_y*n_rois)//(n_rois-1)
    return {'shape': stitched_shape, 'gap': gap}
