import pathlib
from typing import Dict, Tuple, Union
import numpy as np
import tifffile
from skimage.transform import resize as skimage_resize
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.modules.mesoscope_splitting.tiff_splitter import (
    AvgImageTiffSplitter)
from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)


def stitch_tiff_with_rois(
        full_field_path: Union[pathlib.Path, str],
        avg_surface_path: Union[pathlib.Path, str]) -> np.ndarray:
    """
    Create the stitched full field image with ROIs from the
    average surface TIFF inserted

    Parameters
    ----------
    full_field_path: Union[pathlib.Path, str]
        Path to the full field TIFF file

    avg_surface_path: Union[pathlib.Path, str]
        Path to the averaged surface TIFF file

    Returns
    -------
    stitched_roi_img: np.ndarray
    """

    if isinstance(full_field_path, str):
        full_field_path = pathlib.Path(full_field_path)

    if isinstance(avg_surface_path, str):
        avg_surface_path = pathlib.Path(avg_surface_path)

    full_field_img = stitch_full_field_tiff(full_field_path)
    full_field_metadata = ScanImageMetadata(full_field_path)
    avg_splitter = AvgImageTiffSplitter(avg_surface_path)

    stitched_roi_img = _insert_rois_into_surface_img(
            full_field_img=full_field_img,
            full_field_metadata=full_field_metadata,
            avg_image_splitter=avg_splitter)

    return stitched_roi_img


def stitch_full_field_tiff(
       tiff_path: pathlib.Path) -> np.ndarray:
    """
    Create the stitched version of the full-field average
    image from a full-field TIFF file.

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the full field TIFF file

    Returns
    -------
    stitched_img: np.ndarray
        The full field image as an array of np.uint16
    """

    img = _average_full_field_tiff(tiff_path)
    tiff_metadata = ScanImageMetadata(tiff_path)
    img = _stitch_full_field_tiff(
            tiff_metadata=tiff_metadata,
            avg_img=img)
    return normalize_array(array=img, dtype=np.uint16)


def _insert_rois_into_surface_img(
        full_field_img: np.ndarray,
        full_field_metadata: ScanImageMetadata,
        avg_image_splitter: AvgImageTiffSplitter) -> np.ndarray:
    """
    Insert thumbnails from avg_image_splitter into a stitched
    full field image.

    Parameters
    ----------
    full_field_img: np.ndarray
        The stitched full field image into which we are inserting
        the ROI thumbnails

    full_field_metadata: ScanImageMetadata
        Metadata read from the original full_field_img TIFF file

    avg_image_splitter: AvgImageTiffSplitter
        Splitter which will provide the ROI thumbnails (using
        get_avg_img(i_roi=i_roi)

    Returns
    -------
    output_img: np.ndarray
        A copy of full_filed_img with the ROI thumbnails
        from avg_image_splitter superimposed in the correct
        location.

    Notes
    -----
    One of the first steps is ot make a copy of full_field_img,
    so this method does not alter full_field_img in place.
    """

    (ff_resolution,
     ff_size) = _validate_all_roi_same_size(full_field_metadata)

    (origin_col,
     origin_row) = _get_origin(full_field_metadata)

    output_img = np.copy(full_field_img)

    physical_to_pixels = (ff_resolution[0]/ff_size[0],
                          ff_resolution[1]/ff_size[1])

    for i_roi in range(avg_image_splitter.n_rois):
        roi_size = avg_image_splitter.roi_size(i_roi=i_roi)
        roi_resolution = avg_image_splitter.roi_resolution(i_roi=i_roi)
        roi_center = avg_image_splitter.roi_center(i_roi=i_roi)
        avg_img = avg_image_splitter.get_avg_img(i_roi=i_roi, z_value=None)

        roi_conversion = (roi_resolution[0]/roi_size[0],
                          roi_resolution[1]/roi_size[1])

        # physical_to_pixels and roi_conversion are in units of
        # pixels per degree; if the ROI has a larger pixels per
        # degree than the destination image, the ROI has to be
        # downsampled to a smaller pixel grid to fit into the
        # full field image.
        rescaling_factor = (physical_to_pixels[0]/roi_conversion[0],
                            physical_to_pixels[1]/roi_conversion[1])

        avg_img_shape = avg_img.shape

        # remember the difference between XY coordinates and (row, col)
        # coordinates
        new_rows = np.round(avg_img_shape[0]*rescaling_factor[1])
        new_rows = new_rows.astype(int)

        new_cols = np.round(avg_img_shape[1]*rescaling_factor[0])
        new_cols = new_cols.astype(int)

        avg_img = skimage_resize(avg_img,
                                 output_shape=(new_rows, new_cols))

        avg_img = normalize_array(
                        array=avg_img,
                        dtype=np.uint16)

        row0 = roi_center[1]-roi_size[1]/2
        row0 -= origin_row
        row0 = np.round(row0*physical_to_pixels[1]).astype(int)

        col0 = roi_center[0]-roi_size[0]/2
        col0 -= origin_col
        col0 = np.round(col0*physical_to_pixels[0]).astype(int)

        row1 = row0+avg_img.shape[0]
        col1 = col0+avg_img.shape[1]

        output_img[row0:row1,
                   col0:col1] = avg_img

    return output_img


def _average_full_field_tiff(
        tiff_path: pathlib.Path) -> np.ndarray:
    """
    Read in the image data from a full field TIFF image and average
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
        tiff_metadata: ScanImageMetadata,
        avg_img: np.ndarray) -> Dict:
    """
    Get the final shape for the stitched TIFF to be produced
    for a given full field TIFF

    Parameters
    ----------
    tiff_metadata: ScanImageMetadata
        The metadata object associated with this avg_img

    avg_img: np.ndarray
        Average image produced by _average_full_field_tiff

    Returns
    -------
    shape_dict: Dict
        'shape': the final shape of the stitched tiff
        'gap': the gap (in pixels) between columns in the final stitched image
    """

    # Make sure that every ROI only has one scanfield
    for roi in tiff_metadata.defined_rois:
        if not isinstance(roi['scanfields'], dict):
            msg = f"{tiff_metadata.file_path}\n"
            msg += "contains an ROI with more than one scanfield;\n"
            msg += "uncertain how to handle this case"
            raise ValueError(msg)

    # Make sure that every ROI has the same size in pixels as determined
    # by pixelResolutionXY
    resolution = None
    for i_roi in range(len(tiff_metadata.defined_rois)):
        this_resolution = tiff_metadata.roi_resolution(i_roi)
        if resolution is None:
            resolution = this_resolution
        else:
            if resolution != this_resolution:
                msg = f"{tiff_metadata.file_path}\n"
                msg += "contains ROIs with different pixel resolutions;\n"
                msg += "uncertain how to handle this case"
                raise ValueError(msg)

    n_rois = len(tiff_metadata.defined_rois)

    # image coordinates...
    stitched_shape = (resolution[1],
                      resolution[0]*n_rois)

    gap = (avg_img.shape[0] - resolution[1]*n_rois)//(n_rois-1)

    # check that avg_img has expected shape based on this finding
    expected_avg_shape = (n_rois*stitched_shape[0]+(n_rois-1)*gap,
                          stitched_shape[1]//n_rois)

    if avg_img.shape != expected_avg_shape:
        msg = f"{tiff_metadata.file_path}\n"
        msg += "expected average over pages to have shape "
        msg += f"{expected_avg_shape}\n"
        msg += f"got {avg_img.shape}\n"
        msg += "unsure how to proceed with stitching"
        raise ValueError(msg)

    return {'shape': stitched_shape, 'gap': gap}


def _validate_all_roi_same_size(
        metadata: ScanImageMetadata) -> Tuple[Tuple[int,  int],
                                              Tuple[float, float]]:
    """
    Scan through the ROIs in the ScanImageMetadata
    and verify that they all have the same
    sizeXY and pixelResolutionXY

    Returns
    -------
    pixelResolutionXY

    sizeXY
    """

    resolution = None
    physical_size = None

    for i_roi in range(metadata.n_rois):
        this_resolution = metadata.roi_resolution(i_roi)
        this_size = metadata.roi_size(i_roi)
        if resolution is None:
            resolution = this_resolution
            physical_size = this_size
        else:
            if not np.allclose(this_size, physical_size):
                msg = f"{metadata.file_path}\n"
                msg += "has ROIs with different physical units (sizeXY)"
                raise ValueError(msg)
            if not this_resolution == resolution:
                msg = f"{metadata.file_path}\n"
                msg += "has ROIs with different pixel resolutions"
                raise ValueError(msg)

    return (resolution, physical_size)


def _get_origin(
        metadata: ScanImageMetadata) -> Tuple[float, float]:
    """
    Get the XY origin implied by all of the ROIs in a
    TIFF file
    """
    origin_row = None
    origin_col = None
    for i_roi in range(metadata.n_rois):
        roi_center = metadata.roi_center(i_roi=i_roi)
        physical_size = metadata.roi_size(i_roi=i_roi)
        this_row_min = roi_center[1]-physical_size[1]/2
        this_col_min = roi_center[0]-physical_size[0]/2
        if origin_row is None or this_row_min < origin_row:
            origin_row = this_row_min
        if origin_col is None or this_col_min < origin_col:
            origin_col = this_col_min

    return (origin_col, origin_row)


def _stitch_full_field_tiff(
        tiff_metadata: ScanImageMetadata,
        avg_img: np.ndarray) -> np.ndarray:
    """
    Stitch the full field TIFF into a single image, i.e.
    take the image produced by _average_full_field_tiff
    and rearrange its pixels to remove the artificial
    gaps betwen ROIs and arrange the ROIs according
    to their actual positions in physical space.

    Parameters
    ----------
    tiff_metadata: ScanImageMetadata
        The metadata associated with thi avg_image

    avg_img: np.ndarray
        average image returned by _average_full_field_tiff

    Returns
    -------
    stitched_img: np.ndarray
    """

    final_shapes = _get_stitched_tiff_shapes(
            tiff_metadata=tiff_metadata,
            avg_img=avg_img)

    stitched_shape = final_shapes['shape']
    pixel_gap = final_shapes['gap']

    # Make sure ROIs all have the same size in physical
    # units and pixels

    (resolution,
     physical_size) = _validate_all_roi_same_size(tiff_metadata)

    physical_to_pixels = (resolution[0]/physical_size[0],
                          resolution[1]/physical_size[1])

    (origin_col,
     origin_row) = _get_origin(tiff_metadata)

    stitched_img = np.zeros(stitched_shape, dtype=avg_img.dtype)

    for i_roi in range(tiff_metadata.n_rois):
        roi_center = tiff_metadata.roi_center(i_roi=i_roi)
        roi_row0 = roi_center[1]-physical_size[1]/2
        roi_col0 = roi_center[0]-physical_size[0]/2
        pix_row0 = np.round((roi_row0-origin_row)*physical_to_pixels[1])
        pix_row0 = pix_row0.astype(int)
        pix_col0 = np.round((roi_col0-origin_col)*physical_to_pixels[0])
        pix_col0 = pix_col0.astype(int)

        sub_img = avg_img[i_roi*(resolution[1]+pixel_gap):
                          i_roi*pixel_gap+(i_roi+1)*resolution[1],
                          :]

        stitched_img[pix_row0:pix_row0+resolution[1],
                     pix_col0:pix_col0+resolution[0]] = sub_img

    return stitched_img
