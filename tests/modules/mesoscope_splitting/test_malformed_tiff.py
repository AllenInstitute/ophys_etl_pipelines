import pytest
from unittest.mock import patch, Mock
import pathlib
import tempfile

from ophys_etl.modules.mesoscope_splitting.tiff_splitter import (
    ScanImageTiffSplitter)


def test_repeated_z_error(
        tmp_path_factory):
    """
    Test that, if one ROI has the same z more than once,
    an error is raised upon metadata validation
    """
    tmp_dir = tmp_path_factory.mktemp('repeated_z_error')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmp_dir,
                                             suffix='.tiff')[1])

    z_lineup = [[1, 2], [3, 4]]
    roi_list = [{'zs': [1, 1]}, {'zs': [3, 4]}]
    img_grp = {'rois': roi_list}
    roi_grp = {'imagingRoiGroup': img_grp}
    roi_metadata = {'RoiGroups': roi_grp}
    metadata = [{'SI.hStackManager.zsAllActuators': z_lineup,
                 'SI.hChannels.channelSave': [1, 2]},
                roi_metadata]

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):

        with pytest.raises(RuntimeError, match="has duplicate zs"):
            ScanImageTiffSplitter(tiff_path=tmp_path)


def test_roi_order_error(
        tmp_path_factory):
    """
    Test that, if the zs in the SI.hstackManager.zsAllActuators field
    are not listed in ROI order, an error is raised
    """
    tmp_dir = tmp_path_factory.mktemp('repeated_z_error')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmp_dir,
                                             suffix='.tiff')[1])

    z_lineup = [[1, 2], [3, 4]]
    roi_list = [{'zs': [1, 3]}, {'zs': [2, 4]}]
    img_grp = {'rois': roi_list}
    roi_grp = {'imagingRoiGroup': img_grp}
    roi_metadata = {'RoiGroups': roi_grp}
    metadata = [{'SI.hStackManager.zsAllActuators': z_lineup,
                 'SI.hChannels.channelSave': [1, 2]},
                roi_metadata]

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):

        with pytest.raises(RuntimeError,
                           match="not in correct order for ROIs"):
            ScanImageTiffSplitter(tiff_path=tmp_path)


def test_uneven_z_per_roi(
        tmp_path_factory):
    """
    Test that an error is raised if there are not the same number
    of zs per ROI
    """
    tmp_dir = tmp_path_factory.mktemp('repeated_z_error')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmp_dir,
                                             suffix='.tiff')[1])

    z_lineup = [[1, 2, 3], [4, 5]]
    roi_list = [{'zs': [1, 2, 3]}, {'zs': [4, 5]}]
    img_grp = {'rois': roi_list}
    roi_grp = {'imagingRoiGroup': img_grp}
    roi_metadata = {'RoiGroups': roi_grp}
    metadata = [{'SI.hStackManager.zsAllActuators': z_lineup,
                 'SI.hChannels.channelSave': [1, 2]},
                roi_metadata]

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):

        with pytest.raises(RuntimeError,
                           match="equal number of zs per ROI"):
            ScanImageTiffSplitter(tiff_path=tmp_path)


def test_zs_not_list(
        tmp_path_factory):
    """
    Test that an error is raised if zsAllActuators is not a list
    """
    tmp_dir = tmp_path_factory.mktemp('repeated_z_error')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmp_dir,
                                             suffix='.tiff')[1])

    z_lineup = 5
    roi_list = [{'zs': [1, 2, 3]}, {'zs': [4, 5]}]
    img_grp = {'rois': roi_list}
    roi_grp = {'imagingRoiGroup': img_grp}
    roi_metadata = {'RoiGroups': roi_grp}
    metadata = [{'SI.hStackManager.zsAllActuators': z_lineup,
                 'SI.hChannels.channelSave': [1, 2]},
                roi_metadata]

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):

        with pytest.raises(RuntimeError,
                           match="Unclear how to split"):
            ScanImageTiffSplitter(tiff_path=tmp_path)


def test_zs_no_placeholder(
        tmp_path_factory):
    """
    Test that error is raised if the placeholder zeros are missing
    when SI.hChannels.channelSave == 1
    """
    tmp_dir = tmp_path_factory.mktemp('repeated_z_error')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmp_dir,
                                             suffix='.tiff')[1])

    z_lineup = [[5, 4], [6, 7], [8, 9]]
    roi_list = [{'zs': [1, 2, 3]}, {'zs': [4, 5]}]
    img_grp = {'rois': roi_list}
    roi_grp = {'imagingRoiGroup': img_grp}
    roi_metadata = {'RoiGroups': roi_grp}
    metadata = [{'SI.hStackManager.zsAllActuators': z_lineup,
                 'SI.hChannels.channelSave': 1},
                roi_metadata]

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):

        with pytest.raises(RuntimeError,
                           match="channelSave==1"):
            ScanImageTiffSplitter(tiff_path=tmp_path)


@pytest.mark.parametrize("channelSave", [2, 3, [1, 5], [1, 2, 3]])
def test_illegal_channelSave(
        tmp_path_factory,
        channelSave):
    """
    Test that error is raised if channelSave is of an unexpected
    type
    """
    tmp_dir = tmp_path_factory.mktemp('repeated_z_error')
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmp_dir,
                                             suffix='.tiff')[1])

    z_lineup = [[5, 4], [6, 7], [8, 9]]
    roi_list = [{'zs': [1, 2, 3]}, {'zs': [4, 5]}]
    img_grp = {'rois': roi_list}
    roi_grp = {'imagingRoiGroup': img_grp}
    roi_metadata = {'RoiGroups': roi_grp}
    metadata = [{'SI.hStackManager.zsAllActuators': z_lineup,
                 'SI.hChannels.channelSave': channelSave},
                roi_metadata]

    with patch('tifffile.read_scanimage_metadata',
               new=Mock(return_value=metadata)):

        with pytest.raises(RuntimeError,
                           match="Expect channelSave == 1 or"):
            ScanImageTiffSplitter(tiff_path=tmp_path)
