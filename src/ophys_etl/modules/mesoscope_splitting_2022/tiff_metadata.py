from typing import List
import tifffile
import copy
import pathlib


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
        if not tiff_path.is_file():
            raise ValueError(f"{tiff_path.resolve().absolute()} "
                             "is not a file")
        self._metadata = tifffile.read_scanimage_metadata(open(tiff_path, 'rb'))

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
                self._defined_rois = [roi_group]
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

    def zs_for_roi(self, i_roi:int) -> List[int]:
        """
        Return a list of the z-values at which the specified
        ROI was scanned
        """
        if i_roi > self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} "
            msg += "specified in this TIFF file"
            raise ValueError(msg)
        return self.defined_rois[i_roi]['zs']
