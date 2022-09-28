import pytest
from unittest.mock import Mock, patch
import numpy as np
import pathlib
import tifffile
import tempfile
import copy
from itertools import product
from ophys_etl.modules.mesoscope_splitting.full_field_utils import (
    _average_full_field_tiff,
    _get_stitched_tiff_shapes)


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
        Number of rows in the field of view image

    ncols: int
        Number of columns in the field of view image

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
        roiy: int):
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

    Returns
    -------
    roi_metadata: dict
    """

    roi_metadata = {
        'RoiGroups':
            {'imagingRoiGroup': {'rois': list()}}}

    for i_roi in range(nrois):
        this_roi = {'scanfields':
                    {'pixelResolutionXY': [roix, roiy]}}
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
    ncols = 23

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
    assert result['shape'] == (roix*nrois, roiy)

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
                    'scanfields']['pixelResolutionXY'] = (1, 2)

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        with pytest.raises(ValueError, match='different pixel resolutions'):
            _get_stitched_tiff_shapes(
                tiff_path=tiff_path,
                avg_img=avg_img)
