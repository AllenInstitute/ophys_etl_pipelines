from typing import List
import tifffile
import h5py
import pathlib
import numpy as np
from ophys_etl.modules.mesoscope_splitting_2022.tiff_metadata import (
    ScanImageMetadata)


class ZStackSplitter(object):

    def __init__(self, tiff_path_list: List[pathlib.Path]):
        path_to_metadata = dict()
        for tiff_path in tiff_path_list:
            str_path = str(tiff_path.resolve().absolute())
            path_to_metadata[str_path] = ScanImageMetadata(
                                                tiff_path=tiff_path)

        # construct lookup tables to help us map ROI index and z-value
        # to a tiff path and an index in the z-array
        self._roi_z_to_path = dict()
        self._path_z_to_index = dict()
        for tiff_path in path_to_metadata.keys():
            metadata = path_to_metadata[tiff_path]
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
            assert z_mean.shape == (2,)
            if (z_mean % 1).max() > 1.0e-6:
                raise RuntimeError(f"mean z values for {tiff_path} are not "
                                   f"integers: {z_mean}")
            for ii, z_value in enumerate(z_mean):
                self._roi_z_to_path[(this_roi, int(z_value))] = tiff_path
                self._path_z_to_index[(tiff_path, int(z_value))] = ii

        self._path_to_pages = dict()
        for tiff_path in path_to_metadata.keys():
            with tifffile.TiffFile(tiff_path, 'rb') as tiff_file:
                self._path_to_pages[tiff_path] = len(tiff_file.pages)

    def _get_data(self, i_roi: int, z_value: int) -> np.ndarray:
        tiff_path = self._roi_z_to_path[(i_roi, z_value)]
        z_index = self._path_z_to_index[(tiff_path, z_value)]
        data = []
        n_pages = self._path_to_pages[tiff_path]
        with tifffile.TiffFile(tiff_path, 'rb') as tiff_file:
            for i_page in range(z_index, n_pages, 2):
                data.append(tiff_file.pages[i_page].asarray())
        return np.stack(data)

    def write_stack_h5(self,
                       i_roi: int,
                       z_value: int,
                       zstack_path: pathlib.Path) -> None:

        data = self._get_data(i_roi=i_roi, z_value=z_value)
        with h5py.File(zstack_path, 'w') as out_file:
            out_file.create_dataset('data',
                                    data=data,
                                    chunks=(1, data.shape[1], data.shape[2]))
