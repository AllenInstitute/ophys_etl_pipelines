import pytest
from unittest.mock import Mock, patch
import numpy as np
import pathlib
import tifffile
import tempfile
from ophys_etl.modules.mesoscope_splitting.full_field_utils import (
    _average_full_field_tiff)


def _create_full_field_tiff(
        numVolumes: int,
        numSlices: int,
        seed: int,
        output_dir: pathlib.Path):
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

    Returns
    -------
    tiff_path: pathlib.Path
        path to example full field TIFF

    avg_img: np.ndarray
        expected array resulting from taking the average over
        slices and volumes

    metadata:
        Dict containing metadata fields
    """
    rng = np.random.default_rng(seed=seed)
    data = rng.random((numVolumes*numSlices, 17, 21))
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
