import pytest
from unittest.mock import Mock, patch
import numpy as np
import pathlib
import tifffile
import tempfile
import copy
from itertools import product
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)
from ophys_etl.modules.mesoscope_splitting.full_field_utils import (
    _average_full_field_tiff,
    _get_stitched_tiff_shapes,
    _stitch_full_field_tiff,
    _get_origin,
    _validate_all_roi_same_size,
    stitch_full_field_tiff)


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


@pytest.mark.parametrize(
        "numVolumes, numSlices",
        [(17, 13), (1, 1), (5, 1), (1, 27)])
def test_average_full_field_tiff(
        tmpdir_factory,
        helper_functions,
        numVolumes,
        numSlices):
    """
    Test that _average_full_field_tiff properly averages
    over the pages of a full field TIFF image.
    """
    tmpdir = pathlib.Path(
            tmpdir_factory.mktemp('full_field_avg'))

    (tiff_path,
     expected_avg,
     metadata_dict) = _create_full_field_tiff(
                         numVolumes=numVolumes,
                         numSlices=numSlices,
                         seed=213455,
                         output_dir=tmpdir)

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata_dict)):
        actual_avg = _average_full_field_tiff(
                        tiff_path=tiff_path)

    # verify that we are just getting a 2D image out
    assert len(actual_avg.shape) == 2

    np.testing.assert_allclose(actual_avg, expected_avg)
    helper_functions.clean_up_dir(tmpdir=tmpdir)


def test_average_full_field_tiff_failures(
        tmpdir_factory,
        helper_functions):
    """
    Test that expected error is raised when number of pages
    in the full field TIFF is not as expected
    """
    tmpdir = pathlib.Path(
            tmpdir_factory.mktemp('full_field_avg'))

    (tiff_path,
     expected_avg,
     metadata_dict) = _create_full_field_tiff(
                         numVolumes=11,
                         numSlices=13,
                         seed=213455,
                         output_dir=tmpdir)

    wrong_metadata = [{'SI.hStackManager.actualNumVolumes': 5,
                       'SI.hStackManager.actualNumSlices': 2}]

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=wrong_metadata)):

        with pytest.raises(ValueError, match='implies n_pages'):
            _average_full_field_tiff(tiff_path=tiff_path)

    helper_functions.clean_up_dir(tmpdir=tmpdir)


def _create_roi_metadata(
        nrois: int,
        roix: int,
        roiy: int,
        sizex: float = 2.1,
        sizey: float = 3.2):
    """
    Create the dict of ROI metadata for a simulated ScanImage TIFF

    Parameters
    ----------
    nrois: int
        The number of ROIs

    roix: int
        pixelResolutionXY[0] for each ROI

    roiy: int
        pixelResoluitonXY[1] for each ROI

    sizex: float
        sizeXY[0] for each ROI

    sizey: float
        sizeXY[1] for each ROI

    Returns
    -------
    roi_metadata: dict

    Notes
    -----
    ROIs will be given a centerXY value that is the same in y
    but increments in x. This is the arrangement of ROIs in the
    full field TIFF files we are meant to stitch together.
    """

    roi_metadata = {
        'RoiGroups':
            {'imagingRoiGroup': {'rois': list()}}}

    for i_roi in range(nrois):
        this_roi = {'scanfields':
                    {'pixelResolutionXY': [roix, roiy],
                     'sizeXY': [sizex, sizey],
                     'centerXY': [0.5*sizex+i_roi*sizex, 0.5*sizey]}}
        roi_metadata['RoiGroups']['imagingRoiGroup']['rois'].append(this_roi)
    return roi_metadata


@pytest.mark.parametrize(
        "nrois, roiy, roix, gap",
        product((2, 5), (6, 13), (2, 7), (3, 5)))
def test_get_stitched_tiff_shapes(
        tmpdir_factory,
        helper_functions,
        nrois,
        roiy,
        roix,
        gap):
    """
    Test that _get_stitched_tiff_shapes returns the expected
    values for 'shape' and 'gap'
    """

    nrows = gap*(nrois-1)+roiy*nrois
    ncols = roix

    tmpdir = pathlib.Path(
                tmpdir_factory.mktemp('stitched_shapes'))

    (tiff_path,
     avg_img,
     metadata) = _create_full_field_tiff(
                     numVolumes=5,
                     numSlices=3,
                     seed=112358,
                     output_dir=tmpdir,
                     nrows=nrows,
                     ncols=ncols)

    roi_metadata = _create_roi_metadata(
            nrois=nrois,
            roix=roix,
            roiy=roiy)

    metadata.append(roi_metadata)
    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        result = _get_stitched_tiff_shapes(
                    tiff_path=tiff_path,
                    avg_img=avg_img)

    assert result['gap'] == gap
    assert result['shape'] == (roiy, roix*nrois)

    helper_functions.clean_up_dir(tmpdir)


@pytest.mark.parametrize(
        "dcols, drows", [(0, 1), (1, 0)])
def test_get_stitched_tiff_shapes_validation(
        tmpdir_factory,
        helper_functions,
        dcols,
        drows):
    """
    Test that _get_stitched_tiff_shapes throws expected error
    when the average image has the wrong shape

    dcols, drows are integers indicating how much
    the number of rows/columns in the avg_img should be off
    """

    nrois = 7
    roiy = 11
    roix = 13
    gap = 3

    roi_metadata = _create_roi_metadata(
            nrois=nrois,
            roix=roix,
            roiy=roiy)

    tmpdir = pathlib.Path(
                tmpdir_factory.mktemp('stitched_shapes'))

    (tiff_path,
     avg_img,
     metadata) = _create_full_field_tiff(
                     numVolumes=5,
                     numSlices=3,
                     seed=112358,
                     output_dir=tmpdir,
                     nrows=gap*(nrois-1)+roiy*nrois+drows,
                     ncols=roix+dcols)

    metadata.append(roi_metadata)
    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        with pytest.raises(ValueError, match="expected average over pages"):
            _get_stitched_tiff_shapes(
                tiff_path=tiff_path,
                avg_img=avg_img)

    helper_functions.clean_up_dir(tmpdir)


def test_get_stitched_tiff_shapes_errors(
        tmpdir_factory,
        helper_functions):
    """
    Test that _get_stitched_tiff_shapes raises
    errors when the ROI metadata is not as expected
    """

    nrois = 3
    roix = 11
    roiy = 7
    gap = 3
    nrows = gap*(nrois-1)+roiy*nrois
    ncols = 23

    tmpdir = pathlib.Path(
                tmpdir_factory.mktemp('stitched_shapes'))

    (tiff_path,
     avg_img,
     baseline_metadata) = _create_full_field_tiff(
                     numVolumes=5,
                     numSlices=3,
                     seed=112358,
                     output_dir=tmpdir,
                     nrows=nrows,
                     ncols=ncols)

    roi_metadata = _create_roi_metadata(
            nrois=nrois,
            roix=roix,
            roiy=roiy)

    baseline_metadata.append(roi_metadata)

    # if an ROI has more than one scanfield
    metadata = copy.deepcopy(baseline_metadata)
    metadata[1][
        'RoiGroups'][
            'imagingRoiGroup'][
                'rois'][1]['scanfields'] = (1, 2)

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        with pytest.raises(ValueError, match='more than one scanfield'):
            _get_stitched_tiff_shapes(
                tiff_path=tiff_path,
                avg_img=avg_img)

    # if an ROI has different resolution than others
    metadata = copy.deepcopy(baseline_metadata)
    metadata[1][
        'RoiGroups'][
            'imagingRoiGroup'][
                'rois'][1][
                    'scanfields']['pixelResolutionXY'] = (roix, 999)

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        with pytest.raises(ValueError, match='different pixel resolutions'):
            _get_stitched_tiff_shapes(
                tiff_path=tiff_path,
                avg_img=avg_img)

    # if an ROI has different resolution than others
    metadata = copy.deepcopy(baseline_metadata)
    metadata[1][
        'RoiGroups'][
            'imagingRoiGroup'][
                'rois'][1][
                    'scanfields']['pixelResolutionXY'] = (999, roiy)

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        with pytest.raises(ValueError, match='different pixel resolutions'):
            _get_stitched_tiff_shapes(
                tiff_path=tiff_path,
                avg_img=avg_img)


@pytest.mark.parametrize(
        "n_rois, roi_rows, roi_cols, gap",
        [
         (3, 11, 17, 2),
         (2, 27, 31, 1),
         (7, 18, 29, 4)
        ])
def test_stitch_full_field_tiff(
        tmpdir_factory,
        helper_functions,
        n_rois,
        roi_rows,
        roi_cols,
        gap):
    """
    Test that _stitch_full_field_tiff correctly disassembles and
    re-assembles the average image

    n_rois -- an int indicating how many ROIs to simulate in the avg_img

    roi_rows -- an int indicating how many rows are in each ROI

    roi_cols -- an int indicating how many cols are in each ROI

    gap -- an int indicating how many pixels occur between each ROI
    """
    rng = np.random.default_rng(58132134)

    nrows = gap*(n_rois-1)+roi_rows*n_rois
    ncols = roi_cols
    avg_img = np.zeros((nrows, ncols), dtype=float)
    avg_img *= np.NaN

    expected_stitched = np.zeros((roi_rows, n_rois*roi_cols),
                                 dtype=float)

    list_of_rois = []
    for i_roi in range(n_rois):
        r0 = i_roi*(roi_rows+gap)
        r1 = r0+roi_rows
        tile = rng.random((roi_rows, roi_cols))
        list_of_rois.append(tile)
        avg_img[r0:r1, :] = tile
        expected_stitched[:, i_roi*roi_cols:(i_roi+1)*roi_cols] = tile

    metadata = ['nothing',
                _create_roi_metadata(
                    nrois=n_rois,
                    roix=roi_cols,
                    roiy=roi_rows)]

    tmpdir = pathlib.Path(
            tmpdir_factory.mktemp('stitch_full_field'))
    tiff_path = pathlib.Path(
            tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    with patch("tifffile.read_scanimage_metadata",
               new=Mock(return_value=metadata)):

        stitched_img = _stitch_full_field_tiff(
                tiff_path=tiff_path,
                avg_img=avg_img)

    # make sure the gap pixels were all ignored
    assert np.all(np.logical_not(np.isnan(stitched_img)))

    np.testing.assert_allclose(stitched_img, expected_stitched)

    helper_functions.clean_up_dir(tmpdir)


@pytest.mark.parametrize(
        "n_rois, roi_rows, roi_cols, gap",
        [
         (3, 11, 17, 2),
         (2, 27, 31, 1),
         (7, 18, 29, 4)
        ])
def test_user_facing_stitch_full_field_tiff(
        tmpdir_factory,
        helper_functions,
        n_rois,
        roi_rows,
        roi_cols,
        gap):
    """
    Test that stitch_full_field_tiff correctly disassembles and
    re-assembles the average image

    n_rois -- an int indicating how many ROIs to simulate in the avg_img

    roi_rows -- an int indicating how many rows are in each ROI

    roi_cols -- an int indicating how many cols are in each ROI

    gap -- an int indicating how many pixels occur between each ROI
    """
    rng = np.random.default_rng(58132134)

    nrows = gap*(n_rois-1)+roi_rows*n_rois
    ncols = roi_cols
    avg_img = np.zeros((nrows, ncols), dtype=float)
    avg_img *= np.NaN

    expected_stitched = np.zeros((roi_rows, n_rois*roi_cols),
                                 dtype=float)

    list_of_rois = []
    for i_roi in range(n_rois):
        r0 = i_roi*(roi_rows+gap)
        r1 = r0+roi_rows
        tile = rng.random((roi_rows, roi_cols))
        list_of_rois.append(tile)
        avg_img[r0:r1, :] = tile
        expected_stitched[:, i_roi*roi_cols:(i_roi+1)*roi_cols] = tile

    expected_stitched = normalize_array(
            array=expected_stitched,
            dtype=np.uint16)

    metadata = ['nothing',
                _create_roi_metadata(
                    nrois=n_rois,
                    roix=roi_cols,
                    roiy=roi_rows)]

    tmpdir = pathlib.Path(
            tmpdir_factory.mktemp('stitch_full_field'))
    tiff_path = pathlib.Path(
            tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    with patch("tifffile.read_scanimage_metadata",
               new=Mock(return_value=metadata)):
        with patch("ophys_etl.modules.mesoscope_splitting."
                   "full_field_utils._average_full_field_tiff",
                   new=Mock(return_value=avg_img)):
            stitched_img = stitch_full_field_tiff(
                    tiff_path=tiff_path)

    # make sure the gap pixels were all ignored
    assert np.all(np.logical_not(np.isnan(stitched_img)))

    assert stitched_img.dtype == np.uint16

    np.testing.assert_allclose(stitched_img, expected_stitched)

    helper_functions.clean_up_dir(tmpdir)


def test_get_origin(
        tmpdir_factory,
        helper_functions):
    """
    Test that _get_origin actually returns the physical space
    coordinates of the origin implied by a ScanImageMetadata object
    """

    tmpdir = pathlib.Path(
            tmpdir_factory.mktemp('get_origin'))
    tiff_path = pathlib.Path(
            tempfile.mkstemp(
                dir=tmpdir,
                suffix='.tiff')[1])

    metadata = ['nothing',
                dict()]

    roi_metadata = [
        {'scanfields': {
            'centerXY': [1.1, 2.1],
            'sizeXY': [1.5, 0.62]}},
        {'scanfields': {
            'centerXY': [0.5, 3.4],
            'sizeXY': [0.02, 6.0]}}]

    metadata[1] = {
        'RoiGroups': {
             'imagingRoiGroup': {
                 'rois': roi_metadata}}}

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        metadata_obj = ScanImageMetadata(tiff_path)
        actual = _get_origin(metadata_obj)
    expected = [0.35, 0.4]
    np.testing.assert_allclose(expected, actual)

    helper_functions.clean_up_dir(tmpdir)


def test_validate_all_roi_same_size(
        tmpdir_factory,
        helper_functions):

    tmpdir = pathlib.Path(
            tmpdir_factory.mktemp('validate_roi_size'))
    tiff_path = pathlib.Path(
            tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    metadata = ['nothing',
                None]

    # in case where the ROIs do have the same size, verify
    # that _validate_all_roi_same_size does, in fact, return
    # pixelResolutionXY and sizeXY

    roi_metadata = [
        {'scanfields': {
            'sizeXY': [2.1, 3.2],
            'pixelResolutionXY': [45, 33]}},
        {'scanfields': {
            'sizeXY': [2.1, 3.2],
            'pixelResolutionXY': [45, 33]}}]

    metadata[1] = {'RoiGroups': {
                       'imagingRoiGroup': {
                           'rois': roi_metadata}}}
    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        metadata_obj = ScanImageMetadata(tiff_path)
        (resolution,
         size) = _validate_all_roi_same_size(metadata_obj)
        assert resolution == (45, 33)
        np.testing.assert_allclose(size, [2.1, 3.2])

    # test for error in case where sizeXY differs

    roi_metadata = [
        {'scanfields': {
            'sizeXY': [2.11, 3.2],
            'pixelResolutionXY': [45, 33]}},
        {'scanfields': {
            'sizeXY': [2.1, 3.2],
            'pixelResolutionXY': [45, 33]}}]

    metadata[1] = {'RoiGroups': {
                       'imagingRoiGroup': {
                           'rois': roi_metadata}}}
    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        metadata_obj = ScanImageMetadata(tiff_path)
        with pytest.raises(ValueError, match='different physical units'):
            _validate_all_roi_same_size(metadata_obj)

    # test for error in case where pixelResolutionXY differs

    roi_metadata = [
        {'scanfields': {
            'sizeXY': [2.1, 3.2],
            'pixelResolutionXY': [45, 33]}},
        {'scanfields': {
            'sizeXY': [2.1, 3.2],
            'pixelResolutionXY': [46, 33]}}]

    metadata[1] = {'RoiGroups': {
                       'imagingRoiGroup': {
                           'rois': roi_metadata}}}
    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        metadata_obj = ScanImageMetadata(tiff_path)
        with pytest.raises(ValueError, match='different pixel resolutions'):
            _validate_all_roi_same_size(metadata_obj)

    helper_functions.clean_up_dir(tmpdir)