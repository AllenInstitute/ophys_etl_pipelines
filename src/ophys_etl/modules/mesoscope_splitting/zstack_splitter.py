from typing import List, Tuple
import tifffile
import h5py
import pathlib
import json
import numpy as np
from ophys_etl.modules.mesoscope_splitting.mixins import (
    IntFromZMapperMixin)
from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)


class ZStackSplitter(IntFromZMapperMixin):
    """
    Class to handle splitting all of the _local_z_stack.tiff files
    associated with an OPhys session.

    Parameters
    ----------
    tiff_path_list: List[pathlib.Path]
        List of paths to the _local_z_stack.tiff files
        associated with this OPhys session.
    """

    def __init__(self, tiff_path_list: List[pathlib.Path]):

        self._path_to_metadata = dict()
        self._frame_shape = dict()
        for tiff_path in tiff_path_list:
            str_path = str(tiff_path.resolve().absolute())
            self._path_to_metadata[str_path] = ScanImageMetadata(
                                                tiff_path=tiff_path)

            if self._path_to_metadata[str_path].channelSave != [1, 2]:
                raise RuntimeError(
                    f"metadata for {str_path} has channelSave="
                    f"{self._path_to_metadata[str_path].channelSave}\n"
                    "can only handle channelSave==[1, 2]")

        # construct lookup tables to help us map ROI index and z-value
        # to a tiff path and an index in the z-array

        # map (i_roi, z_value) pairs to TIFF file paths
        self._roi_z_int_to_path = dict()

        # map (tiff_file_path, z_value) to the index, i.e.
        # to which scanned z-value *in this TIFF* does the
        # z-value correspond.
        self._path_z_int_to_index = dict()

        # this is an internal lookup table which we will use
        # to validate that every ROI is represented by a
        # z-stack file
        roi_to_path = dict()

        for tiff_path in self._path_to_metadata.keys():
            metadata = self._path_to_metadata[tiff_path]
            this_roi = None
            for i_roi, roi in enumerate(metadata.defined_rois):
                if roi['discretePlaneMode'] == 0:
                    if this_roi is not None:
                        raise RuntimeError("More than one ROI has "
                                           "discretePlaneMode==0 for "
                                           "{tiff_path}")
                    this_roi = i_roi

            if this_roi is None:
                raise RuntimeError("Could not find discretePlaneMode==0 for "
                                   f"{tiff_path}")

            if this_roi not in roi_to_path:
                roi_to_path[this_roi] = []
            roi_to_path[this_roi].append(tiff_path)

            z_array = np.array(metadata.all_zs())
            if z_array.shape[1] != 2:
                raise RuntimeError(f"z_array for {tiff_path} has odd shape\n"
                                   f"{z_array}")

            z_mean = z_array.mean(axis=0)
            for ii, z_value in enumerate(z_mean):
                roi_z = (this_roi, self._int_from_z(z_value=z_value))
                self._roi_z_int_to_path[roi_z] = tiff_path
                path_z = (tiff_path, self._int_from_z(z_value=z_value))
                self._path_z_int_to_index[path_z] = ii

        # check that every ROI has a z-stack file
        for tiff_path in self._path_to_metadata:
            metadata = self._path_to_metadata[tiff_path]
            n_rois = len(metadata.defined_rois)
            if len(roi_to_path) != n_rois:
                msg = (f"There are {n_rois} ROIs; however, only "
                       f"{len(roi_to_path)} of them are represented in the "
                       "local z-stack TIFFS. Here is a mapping from i_roi to "
                       "TIFF paths\n"
                       f"{json.dumps(roi_to_path, indent=2, sort_keys=True)}"
                       "\n\nThis was determined by scanning the z-stack TIFFs "
                       "and noting which ROIs were marked with "
                       "discretePlaneMode==0")
                raise RuntimeError(msg)

        self._path_to_pages = dict()
        for tiff_path in self._path_to_metadata.keys():
            with tifffile.TiffFile(tiff_path, mode='rb') as tiff_file:
                self._path_to_pages[tiff_path] = len(tiff_file.pages)

    @property
    def input_path(self) -> List[str]:
        """
        The list of files this splitter is trying to split
        """
        output = list(self._path_to_metadata.keys())
        output.sort()
        return output

    def roi_center(self, i_roi: int) -> Tuple[float, float]:
        """
        Return the (X, Y) center coordinates for the ROI specified
        by i_roi.
        """
        center_tol = 1.0e-5
        possible_center = []
        for pair in self._roi_z_int_to_path:
            if pair[0] != i_roi:
                continue
            tiff_path = self._roi_z_int_to_path[pair]
            metadata = self._path_to_metadata[tiff_path]
            possible_center.append(metadata.roi_center(i_roi=i_roi))

        baseline_center = possible_center[0]
        for ii in range(1, len(possible_center)):
            center = possible_center[ii]
            dsq = ((center[0]-baseline_center[0])**2
                   + (center[1]-baseline_center[1])**2)
            if dsq > center_tol:
                msg = "Cannot find consistent center for ROI "
                msg += f"{i_roi}"
        return baseline_center

    def frame_shape(self, i_roi: int, z_value: float) -> Tuple[int, int]:
        """
        Return the (nrows, ncolumns) shape of the first page associated
        with the specified (i_roi, z_value) pair
        """
        roi_z = (i_roi, self._int_from_z(z_value=z_value))
        tiff_path = self._roi_z_int_to_path[roi_z]

        path_z = (tiff_path, self._int_from_z(z_value=z_value))
        z_index = self._path_z_int_to_index[path_z]

        with tifffile.TiffFile(tiff_path, mode='rb') as tiff_file:
            page = tiff_file.pages[z_index].asarray()
        return page.shape

    def _get_tiff_path(self, i_roi: int, z_value: float) -> pathlib.Path:
        """
        Return the tiff path corresponding to the (i_roi, z_value) pair
        """
        roi_z = (i_roi, self._int_from_z(z_value=z_value))
        tiff_path = self._roi_z_int_to_path[roi_z]
        return tiff_path

    def _get_pages(self, i_roi: int, z_value: float) -> np.ndarray:
        """
        Get all of the TIFF pages associated in this z-stack set with
        an (i_roi, z_value) pair. Return as a numpy array shaped like
        (n_pages, nrows, ncolumns)
        """
        tiff_path = self._get_tiff_path(i_roi=i_roi, z_value=z_value)

        path_z = (tiff_path, self._int_from_z(z_value=z_value))
        z_index = self._path_z_int_to_index[path_z]

        data = []
        n_pages = self._path_to_pages[tiff_path]
        baseline_shape = self.frame_shape(i_roi=i_roi, z_value=z_value)
        with tifffile.TiffFile(tiff_path, mode='rb') as tiff_file:
            data = [tiff_file.pages[i_page].asarray()
                    for i_page in range(z_index, n_pages, 2)]

        for this_page in data:
            if this_page.shape != baseline_shape:
                msg = f"ROI {i_roi} z_value {z_value} "
                msg += "give inconsistent page shape"
                raise RuntimeError(msg)

        return np.stack(data)

    def write_output_file(self,
                          i_roi: int,
                          z_value: float,
                          output_path: pathlib.Path) -> None:
        """
        Write the z-stack for a specific ROI, z pair to an
        HDF5 file

        Parameters
        ----------
        i_roi: int
            index of the ROI

        z_value: int
            depth of the plane whose z-stack we are writing

        output_path: pathlib.Path
            path to the HDF5 file to be written

        Returns
        -------
        None
            output is written to the HDF5 file.
        """

        if output_path.suffix != '.h5':
            msg = "expected HDF5 output path; "
            msg += f"you gave {output_path.resolve().absolute()}"
            raise ValueError(msg)

        data = self._get_pages(i_roi=i_roi, z_value=z_value)

        metadata = self._path_to_metadata[
                        self._get_tiff_path(
                            i_roi=i_roi,
                            z_value=z_value)].raw_metadata

        with h5py.File(output_path, 'w') as out_file:
            out_file.create_dataset(
                    'scanimage_metadata',
                    data=json.dumps(metadata).encode('utf-8'))

            out_file.create_dataset('data',
                                    data=data,
                                    chunks=(1, data.shape[1], data.shape[2]))
