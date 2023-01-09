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


def test_raw_metadata(
        tmp_path_factory,
        mock_2x3_metadata_zs,
        mock_2x3_metadata_zsAllActuators,
        mock_1x3_metadata_zsAllActuators,
        mock_3x1_metadata_zsAllActuators):
    """
    Test that ScanImageMetadata.raw_metadata returns the expected result
    """

    tmpdir = tmp_path_factory.mktemp('test_all_zs')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    to_replace = 'ophys_etl.modules.mesoscope_splitting.'
    to_replace += 'tiff_metadata._read_metadata'

    for metadata_fixture in (mock_2x3_metadata_zs[0],
                             mock_2x3_metadata_zsAllActuators[0],
                             mock_1x3_metadata_zsAllActuators[0],
                             mock_3x1_metadata_zsAllActuators[0]):

        expected = metadata_fixture

        with patch(to_replace,
                   new=Mock(return_value=metadata_fixture)):
            metadata = ScanImageMetadata(tiff_path=tmp_path)
            assert expected == metadata.raw_metadata


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


@pytest.mark.parametrize(
        "wrong_numVolumes",
        [2.1, (1, 2, 3), [1, 2, 3], 'abcde'])
def test_numVolumes_errors(
        wrong_numVolumes,
        tmpdir_factory,
        helper_functions):
    """
    Test that errors are raised when
    SI.hStackManager.actualNumVolumes is not an int
    """
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('numVolumes_errors'))
    tiff_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmpdir, suffix='tiff')[1])

    metadata = [{'SI.hStackManager.actualNumVolumes': wrong_numVolumes}]
    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        obj = ScanImageMetadata(tiff_path=tiff_path)
        with pytest.raises(ValueError, match='expected int'):
            obj.numVolumes

    helper_functions.clean_up_dir(tmpdir=tmpdir)


@pytest.mark.parametrize(
        "wrong_numSlices",
        [2.1, (1, 2, 3), [1, 2, 3], 'abcde'])
def test_numSlices_errors(
        wrong_numSlices,
        tmpdir_factory,
        helper_functions):
    """
    Test that errors are raised when
    SI.hStackManager.actualNumSlices is not an int
    """
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('numSlices_errors'))
    tiff_path = pathlib.Path(
                    tempfile.mkstemp(dir=tmpdir, suffix='tiff')[1])

    metadata = [{'SI.hStackManager.actualNumSlices': wrong_numSlices}]
    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        obj = ScanImageMetadata(tiff_path=tiff_path)
        with pytest.raises(ValueError, match='expected int'):
            obj.numSlices

    helper_functions.clean_up_dir(tmpdir=tmpdir)


def test_roi_size(
        tmpdir_factory,
        helper_functions):
    """
    Test that ScanImageMetadata.roi_size returns expected values
    """
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('roi_size'))
    tmp_path = pathlib.Path(
        tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    roi_to_size = [
        (10.1, 2.3),
        (4.5, 6.1),
        (8.3, 2.5)
    ]

    roi_list = [
        {'scanfields': {'sizeXY': roi_to_size[0]}},
        {'scanfields': [{'sizeXY': roi_to_size[1]},
                        {'sizeXY': roi_to_size[1]}]},
        {'scanfields': {'sizeXY': roi_to_size[2]}}]

    metadata = ['nothing', dict()]
    metadata[1]['RoiGroups'] = {
        'imagingRoiGroup': {
            'rois': roi_list}}

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
    for ii in range(3):
        assert tiff_metadata.roi_size(ii) == roi_to_size[ii]

    with pytest.raises(ValueError, match="there are only 3"):
        tiff_metadata.roi_size(3)

    helper_functions.clean_up_dir(tmpdir)


def test_roi_size_errors(
        tmpdir_factory,
        helper_functions):
    """
    Test that ScanImageMetadata.roi_size returns errors when scanfields
    have inconsistent sizes
    """
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('roi_size'))
    tmp_path = pathlib.Path(
        tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    roi_to_size = [
        (10.1, 2.3),
        (4.5, 6.1),
        (8.3, 2.5)
    ]

    # test for error when an ROI has two conflicting values
    # for scanfields:sizeXY

    roi_list = [
        {'scanfields': {'sizeXY': roi_to_size[0]}},
        {'scanfields': [{'sizeXY': roi_to_size[1]},
                        {'sizeXY': roi_to_size[2]}]},
        {'scanfields': {'sizeXY': roi_to_size[2]}}]

    metadata = ['nothing', dict()]
    metadata[1]['RoiGroups'] = {
        'imagingRoiGroup': {
            'rois': roi_list}}

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
        with pytest.raises(ValueError, match="differing sizeXY"):
            tiff_metadata.roi_size(1)

    # test for error when an ROI's scanfields are
    # of the wrong datatype

    roi_list = [
        {'scanfields': {'sizeXY': roi_to_size[0]}},
        {'scanfields': 'this is just a string, huh?'},
        {'scanfields': {'sizeXY': roi_to_size[2]}}]

    metadata = ['nothing', dict()]
    metadata[1]['RoiGroups'] = {
        'imagingRoiGroup': {
            'rois': roi_list}}

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
        with pytest.raises(RuntimeError, match="either a list or a dict"):
            tiff_metadata.roi_size(1)

    helper_functions.clean_up_dir(tmpdir)


def test_roi_resolution(
        tmpdir_factory,
        helper_functions):
    """
    Test that ScanImageMetadata.roi_resolution returns expected values
    """
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('roi_resolution'))
    tmp_path = pathlib.Path(
        tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    roi_to_res = [
        (10, 2),
        (4, 6),
        (8, 2)
    ]

    roi_list = [
        {'scanfields': {'pixelResolutionXY': roi_to_res[0]}},
        {'scanfields': [{'pixelResolutionXY': roi_to_res[1]},
                        {'pixelResolutionXY': roi_to_res[1]}]},
        {'scanfields': {'pixelResolutionXY': roi_to_res[2]}}]

    metadata = ['nothing', dict()]
    metadata[1]['RoiGroups'] = {
        'imagingRoiGroup': {
            'rois': roi_list}}

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
    for ii in range(3):
        assert tiff_metadata.roi_resolution(ii) == roi_to_res[ii]

    with pytest.raises(ValueError, match="there are only 3"):
        tiff_metadata.roi_size(3)

    helper_functions.clean_up_dir(tmpdir)


def test_roi_resolution_errors(
        tmpdir_factory,
        helper_functions):
    """
    Test that ScanImageMetadata.roi_size returns errors when scanfields
    have inconsistent sizes
    """
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('roi_resolution'))
    tmp_path = pathlib.Path(
        tempfile.mkstemp(dir=tmpdir, suffix='.tiff')[1])

    roi_to_res = [
        (10, 2),
        (4, 6),
        (8, 2)
    ]

    # test for error when an ROI has two conflicting values
    # for scanfields:pixelResolutionXY

    roi_list = [
        {'scanfields': {'pixelResolutionXY': roi_to_res[0]}},
        {'scanfields': [{'pixelResolutionXY': roi_to_res[1]},
                        {'pixelResolutionXY': roi_to_res[2]}]},
        {'scanfields': {'pixelResolutionXY': roi_to_res[2]}}]

    metadata = ['nothing', dict()]
    metadata[1]['RoiGroups'] = {
        'imagingRoiGroup': {
            'rois': roi_list}}

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
        with pytest.raises(ValueError, match="differing pixelResolutionXY"):
            tiff_metadata.roi_resolution(1)

    # test for error when an ROI's scanfields are
    # of the wrong datatype

    roi_list = [
        {'scanfields': {'sizeXY': roi_to_res[0]}},
        {'scanfields': 'this is just a string, huh?'},
        {'scanfields': {'sizeXY': roi_to_res[2]}}]

    metadata = ['nothing', dict()]
    metadata[1]['RoiGroups'] = {
        'imagingRoiGroup': {
            'rois': roi_list}}

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):
        tiff_metadata = ScanImageMetadata(tiff_path=tmp_path)
        with pytest.raises(RuntimeError, match="either a list or a dict"):
            tiff_metadata.roi_resolution(1)

    helper_functions.clean_up_dir(tmpdir)
