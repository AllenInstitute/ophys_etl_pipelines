from typing import List, Tuple
import tifffile
import h5py
import pathlib
import numpy as np
from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)


class ZStackSplitter(object):
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

        # construct lookup tables to help us map ROI index and z-value
        # to a tiff path and an index in the z-array

        # map (i_roi, z_value) pairs to TIFF file paths
        self._roi_z_to_path = dict()

        # map (tiff_file_path, z_value) to the index, i.e.
        # to which scanned z-value *in this TIFF* does the
        # z-value correspond.
        self._path_z_to_index = dict()

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

            z_array = np.array(metadata.all_zs())
            if z_array.shape[1] != 2:
                raise RuntimeError(f"z_array for {tiff_path} has odd shape\n"
                                   f"{z_array}")

            z_mean = z_array.mean(axis=0)
            if (z_mean % 1).max() > 1.0e-6:
                raise RuntimeError(f"mean z values for {tiff_path} are not "
                                   f"integers: {z_mean}")
            for ii, z_value in enumerate(z_mean):
                self._roi_z_to_path[(this_roi, int(z_value))] = tiff_path
                self._path_z_to_index[(tiff_path, int(z_value))] = ii

        self._path_to_pages = dict()
        for tiff_path in self._path_to_metadata.keys():
            with tifffile.TiffFile(tiff_path, 'rb') as tiff_file:
                self._path_to_pages[tiff_path] = len(tiff_file.pages)

    def roi_center(self, i_roi: int) -> Tuple[float, float]:
        """
        Return the (X, Y) center coordinates for the ROI specified
        by i_roi.
        """
        center_tol = 1.0e-5
        possible_center = []
        for pair in self._roi_z_to_path:
            if pair[0] != i_roi:
                continue
            tiff_path = self._roi_z_to_path[pair]
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

    def frame_shape(self, i_roi: int, z_value: int) -> Tuple[int, int]:
        """
        Return the (nrows, ncolumns) shape of the first page associated
        with the specified (i_roi, z_value) pair
        """
        tiff_path = self._roi_z_to_path[(i_roi, z_value)]
        z_index = self._path_z_to_index[(tiff_path, z_value)]
        with tifffile.TiffFile(tiff_path, 'rb') as tiff_file:
            page = tiff_file.pages[z_index].asarray()
        return page.shape

    def _get_pages(self, i_roi: int, z_value: int) -> np.ndarray:
        """
        Get all of the TIFF pages associated in this z-stack set with
        an (i_roi, z_value) pair. Return as a numpy array shaped like
        (n_pages, nrows, ncolumns)
        """
        tiff_path = self._roi_z_to_path[(i_roi, z_value)]
        z_index = self._path_z_to_index[(tiff_path, z_value)]
        data = []
        n_pages = self._path_to_pages[tiff_path]
        baseline_shape = self.frame_shape(i_roi=i_roi, z_value=z_value)
        with tifffile.TiffFile(tiff_path, 'rb') as tiff_file:
            for i_page in range(z_index, n_pages, 2):
                this_page = tiff_file.pages[i_page].asarray()
                if this_page.shape != baseline_shape:
                    msg = f"ROI {i_roi} z_value {z_value} "
                    msg += "give inconsistent page shape"
                    raise RuntimeError(msg)
                data.append(this_page)
        return np.stack(data)

    def write_stack_h5(self,
                       i_roi: int,
                       z_value: int,
                       zstack_path: pathlib.Path) -> None:
        """
        Write the z-stack for a specific ROI, z pair to an
        HDF5 file

        Parameters
        ----------
        i_roi: int
            index of the ROI

        z_value: int
            depth of the plane whose z-stack we are writing

        z_stack_path: pathlib.Path
            path to the HDF5 file to be written

        Returns
        -------
        None
            output is written to the HDF5 file.
        """

        data = self._get_pages(i_roi=i_roi, z_value=z_value)
        with h5py.File(zstack_path, 'w') as out_file:
            out_file.create_dataset('data',
                                    data=data,
                                    chunks=(1, data.shape[1], data.shape[2]))
