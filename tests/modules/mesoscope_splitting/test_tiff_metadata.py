import pytest
import tempfile
import copy
from unittest.mock import patch, Mock
import pathlib
from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)


@pytest.fixture(scope='session')
def mock_2x3_metadata_zsAllActuators():
    """
    Return a list representing everything we need from
    the metadata for 2 ROIs each sampled at 3 depths

    (use SI.hStackManager.zsAllActuators)

    Return
    ------
    metadata
    expected breakdown of all_zs
    expected result of defined_rois
    """

    z_breakdown = [[1, 4, 2], [9, 5, 11], [2, 6, 1]]

    si_metadata = dict()
    si_metadata['SI.hStackManager.zsAllActuators'] = z_breakdown

    roi_metadata = dict()
    img_group = dict()
    rois = [{'zs': [1, 2, 4]},
            {'zs': [5, 9, 11]},
            {'zs': [1, 2, 6]}]
    img_group['rois'] = rois
    roi_metadata['RoiGroups'] = {'imagingRoiGroup': img_group}

    metadata = [si_metadata, roi_metadata]
    return (metadata,
            copy.deepcopy(z_breakdown),
            copy.deepcopy(rois))


@pytest.fixture(scope='session')
def mock_2x3_metadata_zs():
    """
    Return a list representing everything we need from
    the metadata for 2 ROIs each sampled at 3 depths

    (use SI.hStackManager.zs)
    """

    z_breakdown = [[1, 4, 2], [9, 5, 11], [2, 6, 1]]

    si_metadata = dict()
    si_metadata['SI.hStackManager.zs'] = z_breakdown

    roi_metadata = dict()
    img_group = dict()
    rois = [{'zs': [1, 2, 4]},
            {'zs': [5, 9, 11]},
            {'zs': [1, 2, 6]}]
    img_group['rois'] = rois
    roi_metadata['RoiGroups'] = {'imagingRoiGroup': img_group}

    metadata = [si_metadata, roi_metadata]
    return (metadata,
            copy.deepcopy(z_breakdown),
            copy.deepcopy(rois))


@pytest.fixture(scope='session')
def mock_2x3_metadata_nozs():
    """
    Return a list representing everything we need from
    the metadata for 2 ROIs each sampled at 3 depths

    (use invalid key for SI.hStackManager.zsAllActuators)
    """

    z_breakdown = [[1, 4, 2], [9, 5, 11], [2, 6, 1]]

    si_metadata = dict()
    si_metadata['SI.hStackManager.silly'] = z_breakdown

    roi_metadata = dict()
    img_group = dict()
    rois = [{'zs': [1, 2, 4]},
            {'zs': [5, 9, 11]},
            {'zs': [1, 2, 6]}]
    img_group['rois'] = rois
    roi_metadata['RoiGroups'] = {'imagingRoiGroup': img_group}

    metadata = [si_metadata, roi_metadata]
    return metadata


@pytest.fixture(scope='session')
def mock_3x1_metadata_zsAllActuators():
    """
    Return a list representing everything we need from
    the metadata for 3 ROIs each sampled at 1 depth

    (use SI.hStackManager.zsAllActuators)
    """

    z_breakdown = [[1, 0], [9, 0], [2, 0]]

    si_metadata = dict()
    si_metadata['SI.hStackManager.zsAllActuators'] = z_breakdown

    roi_metadata = dict()
    img_group = dict()
    rois = [{'zs': 1},
            {'zs': 9},
            {'zs': 2}]
    img_group['rois'] = rois
    roi_metadata['RoiGroups'] = {'imagingRoiGroup': img_group}

    metadata = [si_metadata, roi_metadata]
    return (metadata,
            copy.deepcopy(z_breakdown),
            copy.deepcopy(rois))


@pytest.fixture(scope='session')
def mock_1x3_metadata_zsAllActuators():
    """
    Return a list representing everything we need from
    the metadata for 1 ROI each sampled at 3 depths

    (use SI.hStackManager.zsAllActuators)
    """

    z_breakdown = [[1, 2, 9]]

    si_metadata = dict()
    si_metadata['SI.hStackManager.zsAllActuators'] = z_breakdown

    roi_metadata = dict()
    img_group = dict()
    rois = {'zs': [1, 2, 9]}
    img_group['rois'] = rois
    roi_metadata['RoiGroups'] = {'imagingRoiGroup': img_group}

    metadata = [si_metadata, roi_metadata]
    return (metadata,
            copy.deepcopy(z_breakdown),
            [copy.deepcopy(rois), ])


def test_no_file_error():
    """
    Test that, ScanImageMetadata raises an error if you
    give it a path that doesn't point to a file
    """
    dummy_path = pathlib.Path('not_a_file.tiff')
    with pytest.raises(ValueError, match="is not a file"):
        ScanImageMetadata(tiff_path=dummy_path)


def test_all_zs(
        tmp_path_factory,
        mock_2x3_metadata_zs,
        mock_2x3_metadata_zsAllActuators,
        mock_1x3_metadata_zsAllActuators,
        mock_3x1_metadata_zsAllActuators):
    """
    Test that ScanImageMetadata.all_zs() returns the expected result
    """

    tmpdir = tmp_path_factory.mktemp('test_all_zs')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    expected = [[1, 4, 2], [9, 5, 11], [2, 6, 1]]

    to_replace = 'ophys_etl.modules.mesoscope_splitting.'
    to_replace += 'tiff_metadata._read_metadata'

    for metadata_fixture in (mock_2x3_metadata_zs,
                             mock_2x3_metadata_zsAllActuators,
                             mock_1x3_metadata_zsAllActuators,
                             mock_3x1_metadata_zsAllActuators):

        expected = metadata_fixture[1]

        with patch(to_replace,
                   new=Mock(return_value=metadata_fixture[0])):
            metadata = ScanImageMetadata(tiff_path=tmp_path)
            assert expected == metadata.all_zs()


def test_all_zs_error(
        tmp_path_factory,
        mock_2x3_metadata_nozs):
    """
    Test that expected exception is thrown when metadata lacks both
    SI.hStackManager.zs and SI.hStackManager.zsAllActuators
    """

    tmpdir = tmp_path_factory.mktemp('test_all_zs')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    to_replace = 'ophys_etl.modules.mesoscope_splitting.'
    to_replace += 'tiff_metadata._read_metadata'

    with patch(to_replace,
               new=Mock(return_value=mock_2x3_metadata_nozs)):
        metadata = ScanImageMetadata(tiff_path=tmp_path)
    with pytest.raises(ValueError, match="Cannot load all_zs"):
        metadata.all_zs()


def test_zs_for_roi(
        tmp_path_factory,
        mock_2x3_metadata_zs,
        mock_2x3_metadata_zsAllActuators,
        mock_1x3_metadata_zsAllActuators,
        mock_3x1_metadata_zsAllActuators):
    """
    Test behavior of ScanImageMetadata.zs_for_roi
    """
    tmpdir = tmp_path_factory.mktemp('test_all_zs')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    to_replace = 'ophys_etl.modules.mesoscope_splitting.'
    to_replace += 'tiff_metadata._read_metadata'

    for metadata_fixture in (mock_2x3_metadata_zs,
                             mock_2x3_metadata_zsAllActuators,
                             mock_1x3_metadata_zsAllActuators,
                             mock_3x1_metadata_zsAllActuators):

        expected_rois = metadata_fixture[2]

        with patch(to_replace,
                   new=Mock(return_value=metadata_fixture[0])):
            metadata = ScanImageMetadata(tiff_path=tmp_path)
        assert metadata.n_rois == len(expected_rois)
        for i_roi in range(metadata.n_rois):
            assert metadata.zs_for_roi(i_roi) == expected_rois[i_roi]['zs']

        with pytest.raises(ValueError, match="You asked for ROI"):
            metadata.zs_for_roi(metadata.n_rois)


def test_defined_rois(
        tmp_path_factory,
        mock_2x3_metadata_zs,
        mock_2x3_metadata_zsAllActuators,
        mock_3x1_metadata_zsAllActuators,
        mock_1x3_metadata_zsAllActuators):
    """
    Test that ScanImageMetadata.defined_rois returns the expected resuls
    """
    tmpdir = tmp_path_factory.mktemp('test_all_zs')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    to_replace = 'ophys_etl.modules.mesoscope_splitting.'
    to_replace += 'tiff_metadata._read_metadata'

    for metadata_fixture in (mock_2x3_metadata_zs,
                             mock_2x3_metadata_zsAllActuators,
                             mock_1x3_metadata_zsAllActuators,
                             mock_3x1_metadata_zsAllActuators):

        expected_rois = metadata_fixture[2]

        with patch(to_replace,
                   new=Mock(return_value=metadata_fixture[0])):
            metadata = ScanImageMetadata(tiff_path=tmp_path)
            assert metadata.defined_rois == expected_rois
