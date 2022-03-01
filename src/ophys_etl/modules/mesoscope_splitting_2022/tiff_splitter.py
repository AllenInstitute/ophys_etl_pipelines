from typing import List, Set, Tuple
import tifffile
import pathlib
import numpy as np
from ophys_etl.modules.mesoscope_splitting_2022.tiff_metadata import (
    ScanImageMetadata)


class ScanImageTiffSplitter(object):
    """
    A class to naively split up a tiff file by just looping over
    the scanfields in its ROIs

    **this will not work for z-stacks**

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the TIFF file whose metadata we are parsing
    """

    def __init__(self, tiff_path: pathlib.Path):
        self._file_path = tiff_path
        self._metadata = ScanImageMetadata(
                             tiff_path=tiff_path)

        self._validate_z_stack()
        self._get_z_manifest()

    def _validate_z_stack(self):
        """
        Make sure that the zsAllActuators are arranged the
        way we expect, i.e.
        [[roi0_z0, roi0_z1, roi0_z2...],
         [roi1_z0, roi1_z1, roi1_z2...],
         ...
         [roiN_z0, roiN_z1, roiN_z2...]]
        """
        z_value_array = np.array(self._metadata.all_zs()).flatten()
        defined_rois = self._metadata.defined_rois

        z_per_roi = []

        msg = ""

        # check that the same Z value does not appear more than
        # once in the same ROI
        for i_roi, roi in enumerate(defined_rois):
            if isinstance(roi['zs'], list):
                roi_zs = roi['zs']
            else:
                roi_zs = [roi['zs'], ]
            z_set = set(roi_zs)
            if len(z_set) != len(roi_zs):
                msg += f"roi {i_roi} has duplicate zs: {roi['zs']}\n"
            z_per_roi.append(z_set)

        # check that z values in z_array occurr in ROI order
        current_roi = 0
        for z_value in z_value_array:
            if z_value not in z_per_roi[current_roi]:
                if z_value == 0:
                    # it was just a placeholder
                    continue
                current_roi += 1
                if z_value not in z_per_roi[current_roi]:
                    msg += f"z_value {z_value} from sub array "
                    msg += "not in correct order for ROIS; "
                    msg += f"{z_value_array}; "
                    msg += f"{z_per_roi}\n"

        if len(msg) > 0:
            full_msg = "Unclear how to split this TIFF\n"
            full_msg += f"{self._file_path.resolve().absolute()}\n"
            full_msg += f"{msg}"
            raise RuntimeError(full_msg)

    def _get_z_manifest(self):
        local_z_value_list = np.array(self._metadata.all_zs()).flatten()
        defined_rois = self._metadata.defined_rois

        # create a list of sets indicating which z values were actually
        # scanned in the ROI (this will help us parse the placeholder
        # zeros that sometimes get dropped into
        # SI.hStackManager.zsAllActuators

        valid_z_per_roi = []
        for roi in defined_rois:
            this_z_value = roi['zs']
            if isinstance(this_z_value, int):
                this_z_value = [this_z_value, ]
            valid_z_per_roi.append(set(this_z_value))

        self._valid_z_per_roi = valid_z_per_roi
        self._n_valid_zs = 0
        self._n_rois = len(valid_z_per_roi)
        self._roi_z_manifest = []
        ct = 0
        i_roi = 0
        for zz in local_z_value_list:
            if i_roi >= len(valid_z_per_roi):
                break
            if zz in valid_z_per_roi[i_roi]:
                self._roi_z_manifest.append((i_roi, zz))
                self._n_valid_zs += 1
                ct += 1
                if ct == len(valid_z_per_roi[i_roi]):
                    i_roi += 1
                    ct = 0

    @property
    def valid_z_per_roi(self) -> List[Set[int]]:
        return self._valid_z_per_roi

    @property
    def roi_z_manifest(self) -> List[Tuple[int, int]]:
        return self._roi_z_manifest

    @property
    def n_valid_zs(self) -> int:
        return self._n_valid_zs

    @property
    def n_rois(self) -> int:
        return self._n_rois

    @property
    def n_pages(self):
        if not hasattr(self, '_n_pages'):
            with tifffile.TiffFile(self._file_path, 'rb') as tiff_file:
                self._n_pages = len(tiff_file.pages)
        return self._n_pages

    def _get_offset(self, i_roi: int, z_value: int) -> int:
        found_it = False
        n_step_over = 0
        for roi_z_pair in self.roi_z_manifest:
            if roi_z_pair == (i_roi, z_value):
                found_it = True
                break
            n_step_over += 1
        if not found_it:
            msg = f"Could not find stride for {i_roi}, {z_value}\n"
            msg += f"TIFF file {self._file_path.resolve().absolute()}"
            raise ValueError(msg)
        return n_step_over

    def _get_data(self, i_roi: int, z_value: int) -> List[np.ndarray]:
        """
        Get a list of np.ndarrays representing the image data for
        ROI i_roi at specified z_value
        """

        if i_roi >= self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} ROIs "
            msg += f"in {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        if z_value not in self.valid_z_per_roi[i_roi]:
            msg = f"{z_value} is not a valid z value for ROI {i_roi};"
            msg += f"valid z values are {self.valid_z_per_roi[i_roi]}\n"
            msg += f"TIFF file {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        offset = self._get_offset(i_roi=i_roi, z_value=z_value)

        tiff_data = []
        with tifffile.TiffFile(self._file_path, 'rb') as tiff_file:
            for i_page in range(offset, self.n_pages, self.n_valid_zs):
                arr = tiff_file.pages[i_page].asarray()
                tiff_data.append(arr)

        return tiff_data
