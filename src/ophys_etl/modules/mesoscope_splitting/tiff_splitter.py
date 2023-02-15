from typing import List, Tuple, Optional, Dict
import tifffile
import pathlib
import numpy as np
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.modules.mesoscope_splitting.mixins import (
    IntFromZMapperMixin)
from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)
from ophys_etl.modules.mesoscope_splitting.timeseries_utils import (
    split_timeseries_tiff)


class TiffSplitterBase(IntFromZMapperMixin):
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
        self._frame_shape = dict()

    @property
    def raw_metadata(self):
        """
        The ScanImage metadata as a dict
        """
        return self._metadata.raw_metadata

    def _validate_z_stack(self):
        """
        Make sure that the zsAllActuators are arranged the
        way we expect, i.e.
        [[roi0_z0, roi0_z1, roi0_z2..., roi0_zM],
         [roi0_zM+1, roi0_zM+2, ...],
         [roi1_z0, roi1_z1, roi1_z2..., roi1_zM],
         [roi1_zM+1, roi1_zM+2, ...],
         ...
         [roiN_z0, roiN_z1, roiN_z2...]]

        or, in the case of one ROI

        [z0, z1, z2....]
        """
        # check that self._metadata.channelSave is of a form
        # we can process
        if self._metadata.channelSave not in (1, [1, 2]):
            raise RuntimeError(
                "Expect channelSave == 1 or [1, 2]; got "
                f"{self._metadata.channelSave}\n{self._file_path}")

        z_value_array = self._metadata.all_zs()

        # check that z_value_array is a list of lists
        if not isinstance(z_value_array, list):
            msg = "Unclear how to split this TIFF\n"
            msg += f"{self._file_path.resolve().absolute()}\n"
            msg += f"metadata.all_zs {self._metadata.all_zs()}"
            raise RuntimeError(msg)

        if isinstance(z_value_array[0], list):
            z_value_array = np.concatenate(z_value_array)

        # if self._metadata.channelSave == 1, verify that every
        # value of z_value_array is zero, then remove them
        if isinstance(self._metadata.channelSave, int):
            if self._metadata.channelSave != 1:
                raise RuntimeError(
                    "Expect channelSave == 1 or [1, 2]; got "
                    f"{self._metadata.channelSave}\n{self._file_path}")
            for ii in range(1, len(z_value_array), 2):
                if z_value_array[ii] != 0:
                    raise RuntimeError(
                        "channelSave==1 but z values are "
                        f"{z_value_array}; "
                        "expect every other value to be zero\n"
                        f"{self._file_path}")
            z_value_array = z_value_array[::2]
        else:
            valid_channel = isinstance(self._metadata.channelSave, list)
            if valid_channel:
                valid_channel = (self._metadata.channelSave == [1, 2])

            if not valid_channel:
                raise RuntimeError(
                    "Do not know how to handle channelSave=="
                    f"{self._metadata.channelSave}\n{self._file_path}")

        defined_rois = self._metadata.defined_rois

        z_int_per_roi = []

        msg = ""

        # check that the same Z value does not appear more than
        # once in the same ROI
        for i_roi, roi in enumerate(defined_rois):
            if isinstance(roi['zs'], list):
                roi_zs = roi['zs']
            else:
                roi_zs = [roi['zs'], ]
            roi_z_ints = [self._int_from_z(z_value=zz)
                          for zz in roi_zs]
            z_int_set = set(roi_z_ints)
            if len(z_int_set) != len(roi_zs):
                msg += f"roi {i_roi} has duplicate zs: {roi['zs']}\n"
            z_int_per_roi.append(z_int_set)

        # check that z values in z_array occurr in ROI order
        offset = 0
        n_roi = len(z_int_per_roi)
        n_z_per_roi = len(z_value_array)//n_roi

        # check that every ROI has the same number of zs
        if len(z_value_array) % n_roi != 0:
            msg += "There do not appear to be an "
            msg += "equal number of zs per ROI\n"
            msg += f"n_z: {len(z_value_array)} "
            msg += f"n_roi: {len(z_int_per_roi)}\n"

        for roi_z_ints in z_int_per_roi:
            these_z_ints = set([self._int_from_z(z_value=zz)
                                for zz in
                                z_value_array[offset:offset+n_z_per_roi]])

            if these_z_ints != roi_z_ints:
                msg += "z_values from sub array "
                msg += "not in correct order for ROIs; "
                break
            offset += n_z_per_roi

        if len(msg) > 0:
            full_msg = "Unclear how to split this TIFF\n"
            full_msg += f"{self._file_path.resolve().absolute()}\n"
            full_msg += f"{msg}"
            full_msg += f"all_zs {self._metadata.all_zs()}\nfrom rois:\n"
            for roi in self._metadata.defined_rois:
                full_msg += f"zs: {roi['zs']}\n"
            raise RuntimeError(full_msg)

    def _get_z_manifest(self):
        """
        Populate various member objects that help us keep
        track of what z values go with what ROIs in this TIFF
        """
        local_z_value_list = np.array(self._metadata.all_zs()).flatten()
        defined_rois = self._metadata.defined_rois

        # create a list of sets indicating which z values were actually
        # scanned in the ROI (this will help us parse the placeholder
        # zeros that sometimes get dropped into
        # SI.hStackManager.zsAllActuators

        valid_z_int_per_roi = []
        valid_z_per_roi = []
        for roi in defined_rois:
            this_z_value = roi['zs']
            if isinstance(this_z_value, int):
                this_z_value = [this_z_value, ]
            z_as_int = [self._int_from_z(z_value=zz)
                        for zz in this_z_value]
            valid_z_int_per_roi.append(set(z_as_int))
            valid_z_per_roi.append(this_z_value)

        self._valid_z_int_per_roi = valid_z_int_per_roi
        self._valid_z_per_roi = valid_z_per_roi
        self._n_valid_zs = 0
        self._roi_z_int_manifest = []
        ct = 0
        i_roi = 0
        local_z_index_list = [self._int_from_z(zz)
                              for zz in local_z_value_list]
        for zz in local_z_index_list:
            if i_roi >= len(valid_z_int_per_roi):
                break
            if zz in valid_z_int_per_roi[i_roi]:
                roi_z = (i_roi, zz)
                self._roi_z_int_manifest.append(roi_z)
                self._n_valid_zs += 1
                ct += 1
                if ct == len(valid_z_int_per_roi[i_roi]):
                    i_roi += 1
                    ct = 0

    @property
    def input_path(self) -> pathlib.Path:
        """
        The file this splitter is trying to split
        """
        return self._file_path

    def is_z_valid_for_roi(self,
                           i_roi: int,
                           z_value: float) -> bool:
        """
        Is specified z-value valid for the specified ROI
        """
        z_as_int = self._int_from_z(z_value=z_value)
        return z_as_int in self._valid_z_int_per_roi[i_roi]

    @property
    def roi_z_int_manifest(self) -> List[Tuple[int, int]]:
        """
        A list of tuples. Each tuple is a valid
        (roi_index, z_as_int) pair.
        """
        return self._roi_z_int_manifest

    @property
    def n_valid_zs(self) -> int:
        """
        The total number of valid z values associated with this TIFF.
        """
        return self._n_valid_zs

    @property
    def n_rois(self) -> int:
        """
        The number of ROIs in this TIFF
        """
        return self._metadata.n_rois

    def roi_center(self, i_roi: int) -> Tuple[float, float]:
        """
        The (X, Y) center coordinates of roi_index=i_roi
        """
        return self._metadata.roi_center(i_roi=i_roi)

    def roi_size(
            self,
            i_roi: int) -> Tuple[float, float]:
        """
        The physical space size (x, y) of the i_roith ROI
        """
        return self._metadata.roi_size(i_roi=i_roi)

    def roi_resolution(
            self,
            i_roi: int) -> Tuple[int, int]:
        """
        The pixel resolution of the i_roith ROI
        """
        return self._metadata.roi_resolution(i_roi=i_roi)

    def _get_offset(self, i_roi: int, z_value: float) -> int:
        """
        Get the first page associated with the specified
        i_roi, z_value pair.
        """
        found_it = False
        n_step_over = 0
        this_roi_z = (i_roi, self._int_from_z(z_value=z_value))
        for roi_z_pair in self.roi_z_int_manifest:
            if roi_z_pair == this_roi_z:
                found_it = True
                break
            n_step_over += 1
        if not found_it:
            msg = f"Could not find stride for {i_roi}, {z_value}\n"
            msg += f"TIFF file {self._file_path.resolve().absolute()}"
            raise ValueError(msg)
        return n_step_over

    def frame_shape(self,
                    i_roi: int,
                    z_value: Optional[float]) -> Tuple[int, int]:
        """
        Get the shape of the image for a specified ROI at a specified
        z value

        Parameters
        ----------
        i_roi: int
            index of the ROI

        z_value: Optional[float]
            value of z. If None, z_value will be detected automaticall
            (assuming there is no ambiguity)

        Returns
        -------
        frame_shape: Tuple[int, int]
            (nrows, ncolumns)
        """
        if z_value is None:
            z_value = self._get_z_value(i_roi=i_roi)

        key_pair = (i_roi, self._int_from_z(z_value))

        if key_pair not in self._frame_shape:
            offset = self._get_offset(i_roi=i_roi, z_value=z_value)
            with tifffile.TiffFile(self._file_path, mode='rb') as tiff_file:
                page = tiff_file.pages[offset].asarray()
                self._frame_shape[key_pair] = page.shape
        return self._frame_shape[key_pair]


class AvgImageTiffSplitter(TiffSplitterBase):

    @property
    def n_pages(self):
        """
        The number of pages in this TIFF
        """
        if not hasattr(self, '_n_pages'):
            with tifffile.TiffFile(self._file_path, mode='rb') as tiff_file:
                self._n_pages = len(tiff_file.pages)
        return self._n_pages

    def _get_pages(self, i_roi: int, z_value: float) -> List[np.ndarray]:
        """
        Get a list of np.ndarrays representing the pages of image data
        for ROI i_roi at the specified z_value
        """

        if i_roi >= self.n_rois:
            msg = f"You asked for ROI {i_roi}; "
            msg += f"there are only {self.n_rois} ROIs "
            msg += f"in {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        if not self.is_z_valid_for_roi(i_roi=i_roi, z_value=z_value):
            msg = f"{z_value} is not a valid z value for ROI {i_roi};"
            msg += f"valid z values are {self._valid_z_per_roi[i_roi]}\n"
            msg += f"TIFF file {self._file_path.resolve().absolute()}"
            raise ValueError(msg)

        offset = self._get_offset(i_roi=i_roi, z_value=z_value)

        with tifffile.TiffFile(self._file_path, mode='rb') as tiff_file:
            tiff_data = [tiff_file.pages[i_page].asarray()
                         for i_page in
                         range(offset, self.n_pages, self.n_valid_zs)]

        key_pair = (i_roi, z_value)

        for arr in tiff_data:
            if key_pair in self._frame_shape:
                if arr.shape != self._frame_shape[key_pair]:
                    msg = f"ROI {i_roi} z_value {z_value}\n"
                    msg += "yields inconsistent frame shape"
                    raise RuntimeError(msg)
            else:
                self._frame_shape[key_pair] = arr.shape

        return tiff_data

    def _get_z_value(self, i_roi: int) -> float:
        """
        Return the z_value associated with i_roi, assuming
        there is only one. Raises a RuntimeError if there
        is more than one.
        """
        # When splitting surface TIFFs, there's no sensible
        # way to know the z-value ahead of time (whatever the
        # operator enters is just a placeholder). The block
        # of code below will scan for z-values than align with
        # the specified ROI ID and select the correct z value
        # (assuming there is only one)
        possible_z_values = []
        for pair in self.roi_z_int_manifest:
            if pair[0] == i_roi:
                possible_z_values.append(pair[1])
        if len(possible_z_values) > 1:
            msg = f"{len(possible_z_values)} possible z values "
            msg += f"for ROI {i_roi}; must specify one of\n"
            msg += f"{possible_z_values}"
            raise RuntimeError(msg)
        z_value = possible_z_values[0]
        return self._z_from_int(ii=z_value)

    def get_avg_img(self,
                    i_roi: int,
                    z_value: Optional[float]) -> np.ndarray:
        """
        Get the image created by averaging all of the TIFF
        pages associated with an (i_roi, z_value) pair

        Parameters
        ----------
        i_roi: int

        z_value: Optional[int]
            If None, will be detected automatically (assuming there
            is only one)

        Returns
        -------
        np.ndarray
            of floats
        """

        if not hasattr(self, '_avg_img_cache'):
            self._avg_img_cache = dict()

        if z_value is None:
            z_value = self._get_z_value(i_roi=i_roi)

        z_int = self._int_from_z(z_value=z_value)
        pair = (i_roi, z_int)
        if pair not in self._avg_img_cache:
            data = np.array(self._get_pages(i_roi=i_roi, z_value=z_value))
            avg_img = np.mean(data, axis=0)
            self._avg_img_cache[pair] = avg_img

        return np.copy(self._avg_img_cache[pair])

    def write_output_file(self,
                          i_roi: int,
                          z_value: Optional[float],
                          output_path: pathlib.Path) -> None:
        """
        Write the image created by averaging all of the TIFF
        pages associated with an (i_roi, z_value) pair to a TIFF
        file.

        Parameters
        ----------
        i_roi: int

        z_value: Optional[int]
            If None, will be detected automatically (assuming there
            is only one)

        output_path: pathlib.Path
            Path to file to be written

        Returns
        -------
        None
            Output is written to output_path
        """

        if output_path.suffix not in ('.tif', '.tiff'):
            msg = "expected .tiff output path; "
            msg += f"you specified {output_path.resolve().absolute()}"

        avg_img = self.get_avg_img(
                    i_roi=i_roi,
                    z_value=z_value)

        avg_img = normalize_array(array=avg_img,
                                  lower_cutoff=None,
                                  upper_cutoff=None)

        metadata = {'scanimage_metadata': self.raw_metadata}

        tifffile.imwrite(output_path,
                         avg_img,
                         metadata=metadata)
        return None


class TimeSeriesSplitter(TiffSplitterBase):
    """
    A class specifically for splitting timeseries TIFFs

    Parameters
    ----------
    tiff_path: pathlib.Path
        Path to the TIFF file whose metadata we are parsing
    """

    def write_output_files(self,
                           output_path_map: Dict[Tuple[int, float],
                                                 pathlib.Path],
                           tmp_dir: Optional[pathlib.Path] = None,
                           dump_every: int = 1000,
                           logger: Optional[callable] = None) -> None:
        """
        Write all of the pages associated with an
        (i_roi, z_value) pair to an HDF5 file.

        Parameters
        ----------
        output_path_map: Dict[Tuple[int, float], pathlib.Path]
            Dict mapping (i_roi, z_value) pairs to output paths
            where the data for those ROIs should go.

        tmp_dir: Optional[pathlib.Path]
            Directory where temporary files will be written.

        dump_every: int
            Number of frames to store in each temprorary file
            (see Notes)

        logger: Optional[callable]
            Logger which will be invoked with logger.INFO
            by worker methods

        Returns
        -------
        None
            Timeseries for the ROIs are written to the paths
            specified in output_path_map.

            If not specified, will write temporary files into
            the directory where the final files are meant to
            be written.

        Notes
        -----
        Because the only way to get n_pages from a BigTIFF is to
        iterate over its pages, counting, this module works by iterating
        over the pages, splitting the timeseries data into small temp files
        as it goes and keeping track of how many total pages are being written
        for each ROI. After the temp files are written, the temp files are
        gathered together into the final files specified in output_path_map
        and the temporary files are deleted.
        """

        for key_pair in output_path_map:
            output_path = output_path_map[key_pair]
            if output_path.suffix != '.h5':
                msg = "expected HDF5 output path; "
                msg += f"you gave {output_path.resolve().absolute()}"
                raise ValueError(msg)

            i_roi = key_pair[0]
            z_value = key_pair[1]

            if i_roi < 0:
                msg = f"You asked for ROI {i_roi}; "
                msg += "i_roi must be >= 0"
                raise ValueError(msg)

            if i_roi >= self.n_rois:
                msg = f"You asked for ROI {i_roi}; "
                msg += f"there are only {self.n_rois} ROIs "
                msg += f"in {self._file_path.resolve().absolute()}"
                raise ValueError(msg)

            if not self.is_z_valid_for_roi(i_roi=i_roi, z_value=z_value):
                msg = f"{z_value} is not a valid z value for ROI {i_roi};"
                msg += f"valid z values are {self._valid_z_per_roi[i_roi]}\n"
                msg += f"TIFF file {self._file_path.resolve().absolute()}"
                raise ValueError(msg)

        if len(output_path_map) != len(self.roi_z_int_manifest):
            msg = f"you specified paths for {len(output_path_map)} "
            msg += "timeseries files, but the metadata for this "
            msg += "TIFF says it contains "
            msg += f"{len(self.roi_z_int_manifest)}; "
            msg += "we cannot split this file "
            msg += f"({self._file_path})"
            raise ValueError(msg)

        all_roi_z_int = set()
        for key_pair in output_path_map:
            i_roi = key_pair[0]
            z_value = key_pair[1]
            this_pair = (i_roi, self._int_from_z(z_value=z_value))
            all_roi_z_int.add(this_pair)
        for roi_z_pair in self.roi_z_int_manifest:
            if roi_z_pair not in all_roi_z_int:
                raise ValueError(
                    "You did not specify output paths for all "
                    "of the timeseries in "
                    f"{self._file_path}")

        offset_to_path = dict()
        for key_pair in output_path_map:
            i_roi = key_pair[0]
            z_value = key_pair[1]
            offset = self._get_offset(i_roi=i_roi, z_value=z_value)
            if offset in offset_to_path:
                raise RuntimeError(
                    "Same offset occurs twice when splitting "
                    f"{self._file_path}")
            offset_to_path[offset] = output_path_map[key_pair]

        split_timeseries_tiff(
                tiff_path=self._file_path,
                tmp_dir=tmp_dir,
                offset_to_path=offset_to_path,
                dump_every=dump_every,
                logger=logger,
                metadata=self.raw_metadata)

        return None
