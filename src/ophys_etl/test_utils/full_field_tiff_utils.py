import numpy as np
import tifffile
import pathlib
from ophys_etl.utils.tempfile_util import mkstemp_clean


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
            mkstemp_clean(dir=output_dir,
                          prefix='full_field_',
                          suffix='.tiff'))
    tifffile.imwrite(tiff_path, tiff_pages)
    metadata = [{'SI.hStackManager.actualNumVolumes': numVolumes,
                 'SI.hStackManager.actualNumSlices': numSlices}]

    return (tiff_path, avg_img, metadata)


def _create_roi_metadata(
        nrois: int,
        roix: int,
        roiy: int,
        sizex: float = 2.1,
        sizey: float = 3.2,
        origin_x: float = 0.0,
        origin_y: float = 0.0):
    """
    Create the dict of ROI metadata for a simulated ScanImage TIFF

    Parameters
    ----------
    nrois: int
        The number of ROIs

    roix: int
        pixelResolutionXY[0] for each ROI

    roiy: int
        pixelResolutionXY[1] for each ROI

    sizex: float
        sizeXY[0] for each ROI.

    sizey: float
        sizeXY[1] for each ROI

    origin_x, origin_y: float
        origin of ROI coordinate system in physical units

    Returns
    -------
    roi_metadata: dict

    Notes
    -----
    ROIs will be given a centerXY value that is the same in y
    but increments in x. This is the arrangement of ROIs in the
    full field TIFF files we are meant to stitch together.

    sizexy is the physical size of each ROI that makes up the
    stitched image.
    """

    roi_metadata = {
        'RoiGroups':
            {'imagingRoiGroup': {'rois': list()}}}

    for i_roi in range(nrois):
        this_roi = {'scanfields':
                    {'pixelResolutionXY': [roix, roiy],
                     'sizeXY': [sizex, sizey],
                     'centerXY': [origin_x+0.5*sizex+i_roi*sizex,
                                  origin_y+0.5*sizey]}}
        roi_metadata['RoiGroups']['imagingRoiGroup']['rois'].append(this_roi)
    return roi_metadata
