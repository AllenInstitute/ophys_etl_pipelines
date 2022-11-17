from typing import List, Tuple, Union
import tifffile
import copy
import pathlib
import numpy as np


def _read_metadata(tiff_path: pathlib.Path):
    """
    Calls tifffile.read_scanimage_metadata on the specified
    path and returns teh result. This method was factored
    out so that it could be easily mocked in unit tests.
    """
    return tifffile.read_scanimage_metadata(
                        open(tiff_path, 'rb'))


class ScanImageMetadata(object):
    """
    A class to handle reading and parsing the metadata that
    comes with the TIFF files produced by ScanImage

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the TIFF file whose metadata we are parsing
    """

    def __init__(self, tiff_path: pathlib.Path):
        self._file_path = tiff_path
        if not tiff_path.is_file():
            raise ValueError(f"{tiff_path.resolve().absolute()} "
                             "is not a file")
        self._metadata = _read_metadata(tiff_path)

    @property
    def file_path(self) -> pathlib.Path:
        return self._file_path

    @property
    def raw_metadata(self) -> tuple:
        """
        Return a copy of the raw metadata as read by
        tifffile.read_scanimage_metadata.
        """
        return copy.deepcopy(self._metadata)

    @property
    def numVolumes(self) -> int:
        """
        The metadata field representing the number of volumes
        recorded by the rig
        """
        if not hasattr(self, '_numVolumes'):
            value = self._metadata[0]['SI.hStackManager.actualNumVolumes']
            if not isinstance(value, int):
                raise ValueError(
                    f"in {self._file_path}\n"
                    "SI.hStackManager.actualNumVolumes is a "
                    f"{type(value)}; expected int")

            self._numVolumes = value

        return self._numVolumes

    @property
    def numSlices(self) -> int:
        """
        The metadata field representing the number of slices
        recorded by the rig
        """
        if not hasattr(self, '_numSlices'):
            value = self._metadata[0]['SI.hStackManager.actualNumSlices']
            if not isinstance(value, int):
                raise ValueError(
                    f"in {self._file_path}\n"
                    "SI.hStackManager.actualNumSlices is a "
                    f"{type(value)}; expected int")
            self._numSlices = value

        return self._numSlices

    @property
    def channelSave(self) -> Union[int, List[int]]:
        """
        The metadata field representing which channels were saved
        in this TIFF. Either 1 or [1, 2]
        """
        if not hasattr(self, '_channelSave'):
            self._channelSave = self._metadata[0]['SI.hChannels.channelSave']
        return self._channelSave

    @property
    def defined_rois(self) -> List[dict]:
        """
        Get the ROIs defined in this TIFF file

        This is list of dicts, each dict containing the ScanImage
        metadata for a given ROI

        In this context, an ROI is a 3-dimensional volume of the brain
        that was scanned by the microscope.
        """
        if not hasattr(self, '_defined_rois'):
            roi_parent = self._metadata[1]['RoiGroups']
            roi_group = roi_parent['imagingRoiGroup']['rois']
            if isinstance(roi_group, dict):
                self._defined_rois = [roi_group, ]
            elif isinstance(roi_group, list):
                self._defined_rois = roi_group
            else:
                msg = "unable to parse "
                msg += "self._metadata[1]['RoiGroups']"
                msg += "['imagingROIGroup']['rois'] "
                msg += f"of type {type(roi_group)}"
                raise RuntimeError(msg)

        # use copy to make absolutely sure self._defined_rois
        # is not accidentally changed downstream
        return copy.deepcopy(self._defined_rois)

    @property
    def n_rois(self) -> int:
        """
        Number of ROIs defined in the metadata for this TIFF file.
        """
        if not hasattr(self, '_n_rois'):
            self._n_rois = len(self.defined_rois)
        return self._n_rois

    def zs_for_roi(self, i_roi: int) -> List[int]:
        """
        Return a list of the z-values at which the specified
        ROI was scanned
        """
        if i_roi >= self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} "
            msg += "specified in this TIFF file"
            raise ValueError(msg)
        return self.defined_rois[i_roi]['zs']

    def all_zs(self) -> List:
        """
        Return the structure that lists the z-values of all scans divided
        into imaging groups, i.e.

        scanimage_metadata[0]['SI.hStackManager.zsAllActuators']

        (in historical versions of ScanImage, the desired key is actually
        'SI.hStackManager.zs'; this method will try that if
        'zsAllActuators' is not present)
        """
        key_to_use = 'SI.hStackManager.zsAllActuators'
        if key_to_use in self._metadata[0]:
            return self._metadata[0][key_to_use]

        other_key = 'SI.hStackManager.zs'
        if other_key not in self._metadata[0]:
            msg = "Cannot load all_zs from "
            msg += f"{self._file_path.resolve().absolute()}\n"
            msg += f"Neither {key_to_use} nor "
            msg += f"{other_key} present"
            raise ValueError(msg)

        return self._metadata[0][other_key]

    def roi_center(self,
                   i_roi: int,
                   atol: float = 1.0e-5) -> Tuple[float, float]:
        """
        Return the X, Y center of the specified ROI.

        If the scanfields within an ROI have inconsistent values to within
        absolute tolerance atol, raise an error (this is probably allowed
        by ScanImage; I do not think we are ready to handle it, yet).

        Parameters
        ----------
        i_roi: int

        atol: float
            The tolerance in X and Y within which two
            points in (X, Y) space are allowed to be the same

        Returns
        -------
        center: Tuple[float, float]
           (X_coord, Y_coord)
        """
        if i_roi >= self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} "
            msg += "specified in {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        scanfields = self.defined_rois[i_roi]['scanfields']
        if isinstance(scanfields, dict):
            scanfields = [scanfields]
        elif not isinstance(scanfields, list):
            msg = "Expected scanfields to be either a list "
            msg += f"or a dict; instead got {type(scanfields)}"
            raise RuntimeError(msg)
        avg_x = 0.0
        avg_y = 0.0
        for field in scanfields:
            center = field['centerXY']
            avg_x += center[0]
            avg_y += center[1]
        avg_x = avg_x / len(scanfields)
        avg_y = avg_y / len(scanfields)

        is_valid = True
        for field in scanfields:
            center = field['centerXY']
            if abs(center[0]-avg_x) > atol:
                is_valid = False
            if abs(center[1]-avg_y) > atol:
                is_valid = False

        if not is_valid:
            msg = "\nInconsistent scanfield centers:\n"
            for field in scanfields:
                msg += "{field['centerXY']}\n"
            raise RuntimeError(msg)

        return (avg_x, avg_y)

    def roi_size(
            self,
            i_roi: int) -> Tuple[float, float]:
        """
        Return the size in physical units of an ROI. Will raise an error
        if the ROI has multiple scanfields with inconsistent size values.

        Parameters
        ----------
        i_roi: int:
            Index of the ROI whose size is to be returned.

        Returns
        -------
        sizexy: Tuple[float, float]
            This is just the 'sizeXY' element associated with an ROI's
            scanfield metadata.
        """
        if i_roi >= self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} "
            msg += "specified in {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        scanfields = self.defined_rois[i_roi]['scanfields']
        if isinstance(scanfields, dict):
            scanfields = [scanfields]
        elif not isinstance(scanfields, list):
            msg = "Expected scanfields to be either a list "
            msg += f"or a dict; instead got {type(scanfields)}"
            raise RuntimeError(msg)

        size_x = None
        size_y = None
        for this_scanfield in scanfields:
            if size_x is None:
                size_x = this_scanfield['sizeXY'][0]
                size_y = this_scanfield['sizeXY'][1]
            else:
                same_x = np.allclose(size_x, this_scanfield['sizeXY'][0])
                same_y = np.allclose(size_y, this_scanfield['sizeXY'][1])
                if not same_x or not same_y:
                    msg = f"{self._file_path.resolve().absolute()}\n"
                    msg += f"i_roi: {i_roi}\n"
                    msg += "has multiple scanfields with differing sizeXY\n"
                    msg += "asking for roi_size is meaningless"
                    raise ValueError(msg)

        if size_x is None or size_y is None:
            raise ValueError(
                "Could not find sizeXY for "
                f"ROI {i_roi} in {self._file_path.resolve().absolute()}")

        return (size_x, size_y)

    def roi_resolution(
            self,
            i_roi: int) -> Tuple[int, int]:
        """
        Return the size in pixels of an ROI. Will raise an error
        if the ROI has multiple scanfields with inconsistent values.

        Parameters
        ----------
        i_roi: int:
            Index of the ROI whose size is to be returned.

        Returns
        -------
        resolutionxy: Tuple[int, int]
            This is just the 'pixelResolutionXY' element associated with
            an ROI's scanfield metadata.
        """
        if i_roi >= self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} "
            msg += "specified in {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        scanfields = self.defined_rois[i_roi]['scanfields']
        if isinstance(scanfields, dict):
            scanfields = [scanfields]
        elif not isinstance(scanfields, list):
            msg = "Expected scanfields to be either a list "
            msg += f"or a dict; instead got {type(scanfields)}"
            raise RuntimeError(msg)

        pix_x = None
        pix_y = None
        for this_scanfield in scanfields:
            if pix_x is None:
                pix_x = this_scanfield['pixelResolutionXY'][0]
                pix_y = this_scanfield['pixelResolutionXY'][1]
            else:
                same_x = (pix_x == this_scanfield['pixelResolutionXY'][0])
                same_y = (pix_y == this_scanfield['pixelResolutionXY'][1])
                if not same_x or not same_y:
                    msg = f"{self._file_path.resolve().absolute()}\n"
                    msg += f"i_roi: {i_roi}\n"
                    msg += "has multiple scanfields with differing "
                    msg += "pixelResolutionXY\n"
                    msg += "asking for roi_size is meaningless"
                    raise ValueError(msg)

        if pix_x is None or pix_y is None:
            raise ValueError(
                "Could not find pixelResolutionXY for "
                f"ROI {i_roi} in {self._file_path.resolve().absolute()}")

        return (pix_x, pix_y)
