from typing import List, Tuple, Optional
from argschema import ArgSchemaParser
import pathlib
import numpy as np
import time
import json
import h5py

from ophys_etl.modules.mesoscope_splitting.schemas import (
    InputSchema, OutputSchema)

from ophys_etl.modules.mesoscope_splitting.tiff_splitter import (
    AvgImageTiffSplitter,
    TimeSeriesSplitter)

from ophys_etl.modules.mesoscope_splitting.zstack_splitter import (
    ZStackSplitter)

from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)

from ophys_etl.modules.mesoscope_splitting.output_json_sanitizer import (
    get_sanitized_json_data)

from ophys_etl.modules.mesoscope_splitting.full_field_utils import (
    stitch_full_field_tiff,
    _insert_rois_into_surface_img)


def get_valid_roi_centers(
        timeseries_splitter: TimeSeriesSplitter) -> List[Tuple[float, float]]:
    """
    Return a list of all of the valid ROI centers taken from a
    TimeSeriesSplitter
    """
    eps = 0.01  # ROIs farther apart than this are different
    valid_roi_centers = []
    for roi_z_tuple in timeseries_splitter.roi_z_int_manifest:
        roi_center = timeseries_splitter.roi_center(
                              i_roi=roi_z_tuple[0])
        dmin = None
        for other_roi_center in valid_roi_centers:
            distance = np.sqrt((roi_center[0]-other_roi_center[0])**2
                               + (roi_center[1]-other_roi_center[1])**2)
            if dmin is None or distance < dmin:
                dmin = distance
        if dmin is None or dmin > eps:
            valid_roi_centers.append(roi_center)
    return valid_roi_centers


def get_nearest_roi_center(
        this_roi_center: Tuple[float, float],
        valid_roi_centers: List[Tuple[float, float]]) -> int:
    """
    Take a specified ROI center and a list of valid ROI centers,
    return the index in valid_roi_centers that is closest to
    this_roi_center
    """
    dmin = None
    ans = None
    for i_roi, roi in enumerate(valid_roi_centers):
        dist = ((this_roi_center[0]-roi[0])**2
                + (this_roi_center[1]-roi[1])**2)
        if dmin is None or dist < dmin:
            ans = i_roi
            dmin = dist

    if ans is None:
        msg = "Could not find nearest ROI center for\n"
        msg += f"{this_roi_center}\n"
        msg += f"{valid_roi_centers}\n"
        raise RuntimeError(msg)

    return ans


def write_out_stitched_full_field_image(
        path_to_avg_tiff: pathlib.Path,
        path_to_full_field_tiff: pathlib.Path,
        output_path: pathlib.Path):
    """
    Write out the stitched full field image with
    and without ROIs to an HDF5 file

    Parameters
    ----------
    path_to_avg_tiff: pathlib.Path
        Path to the avaraged surface TIFF

    path_to_full_field_tiff: pathlib.Path
        Path to the raw full field TIFF

    output_path: pathlib.Path
        File to be written out
    """

    full_field_metadata = ScanImageMetadata(
            path_to_full_field_tiff)
    full_field_img = stitch_full_field_tiff(
            path_to_full_field_tiff)

    avg_splitter = AvgImageTiffSplitter(
            path_to_avg_tiff)

    with_rois = _insert_rois_into_surface_img(
        full_field_img=full_field_img,
        full_field_metadata=full_field_metadata,
        avg_image_splitter=avg_splitter)

    with h5py.File(output_path, "w") as out_file:

        out_file.create_dataset(
            "stitched_full_field",
            data=full_field_img)

        out_file.create_dataset(
            "stitched_full_field_with_rois",
            data=with_rois)

        out_file.create_dataset(
            "surface_roi_metadata",
            data=json.dumps(avg_splitter.raw_metadata).encode('utf-8'))

        ff_metadata = json.dumps(full_field_metadata.raw_metadata)
        out_file.create_dataset(
            "full_field_metadata",
            data=ff_metadata.encode('utf-8'))


class TiffSplitterCLI(ArgSchemaParser):
    default_schema = InputSchema
    default_output_schema = OutputSchema

    def run(self):
        t0 = time.time()
        output = {"column_stacks": []}
        files_to_record = []

        ts_path = pathlib.Path(self.args['timeseries_tif'])
        timeseries_splitter = TimeSeriesSplitter(tiff_path=ts_path)
        files_to_record.append(ts_path)

        depth_path = pathlib.Path(self.args["depths_tif"])
        depth_splitter = AvgImageTiffSplitter(tiff_path=depth_path)
        files_to_record.append(depth_path)

        surface_path = pathlib.Path(self.args["surface_tif"])
        surface_splitter = AvgImageTiffSplitter(tiff_path=surface_path)
        files_to_record.append(surface_path)

        zstack_path_list = []
        for plane_grp in self.args['plane_groups']:
            zstack_path = pathlib.Path(plane_grp['local_z_stack_tif'])
            zstack_path_list.append(zstack_path)
            files_to_record.append(zstack_path)

        zstack_splitter = ZStackSplitter(tiff_path_list=zstack_path_list)

        ready_to_archive = set()

        # Looking at recent examples of outputs from this queue,
        # I do not think we have honored the 'column_z_stack_tif'
        # entry in the schema for some time now. I find no examples
        # in which this entry of the input.jon is ever populated.
        # I am leaving it here for now to avoid the complication of
        # having to modify the ruby strategy associated with this
        # queue, which is out of scope for the work we have
        # currently committed to.
        for plane_group in self.args["plane_groups"]:
            if "column_z_stack_tif" in plane_group:
                msg = "'column_z_stack_tif' detected in 'plane_groups'; "
                msg += "the TIFF splitting code no longer handles that file."
                self.logger.warning(msg)

        # There are cases where the centers for ROIs are not
        # exact across modalities, so we cannot demand that the
        # ROI centers be the same to within an absolute tolerance.
        # Here we use the timeseries TIFF to assemble a list of all
        # available ROI centers. When splitting the other TIFFs, we
        # will validate them by making sure that the closest
        # valid_roi_center is always what we expect.
        valid_roi_centers = get_valid_roi_centers(
                                timeseries_splitter=timeseries_splitter)

        experiment_metadata = []
        for plane_group in self.args["plane_groups"]:
            for experiment in plane_group["ophys_experiments"]:
                this_exp_metadata = dict()
                exp_id = experiment["experiment_id"]
                this_exp_metadata["experiment_id"] = exp_id
                for file_key in ('timeseries',
                                 'depth_2p',
                                 'surface_2p',
                                 'local_z_stack'):
                    this_metadata = dict()
                    for data_key in ('offset_x',
                                     'offset_y',
                                     'rotation',
                                     'resolution'):
                        this_metadata[data_key] = experiment[data_key]
                    this_exp_metadata[file_key] = this_metadata

                experiment_dir = pathlib.Path(experiment["storage_directory"])
                experiment_id = experiment["experiment_id"]
                roi_index = experiment["roi_index"]
                scanfield_z = experiment["scanfield_z"]
                baseline_center = None

                for (splitter,
                     z_value,
                     output_name,
                     metadata_tag) in zip(
                                  (depth_splitter,
                                   surface_splitter,
                                   zstack_splitter),
                                  (scanfield_z,
                                   None,
                                   scanfield_z,
                                   scanfield_z),
                                  (f"{experiment_id}_depth.tif",
                                   f"{experiment_id}_surface.tif",
                                   f"{experiment_id}_z_stack_local.h5"),
                                  ("depth_2p",
                                   "surface_2p",
                                   "local_z_stack")):

                    output_path = experiment_dir / output_name

                    roi_center = splitter.roi_center(i_roi=roi_index)
                    nearest_valid = get_nearest_roi_center(
                                        this_roi_center=roi_center,
                                        valid_roi_centers=valid_roi_centers)
                    if baseline_center is None:
                        baseline_center = nearest_valid

                    if nearest_valid != baseline_center:
                        msg = f"experiment {experiment_id}\n"
                        msg += "roi center inconsistent for "
                        msg += "input: "
                        msg += f"{splitter.input_path.resolve().absolute()}\n"
                        msg += "output: "
                        msg += f"{output_path.resolve().absolute()}\n"
                        msg += f"{baseline_center}; {roi_center}\n"
                        raise RuntimeError(msg)

                    splitter.write_output_file(
                                    i_roi=roi_index,
                                    z_value=z_value,
                                    output_path=output_path)
                    str_path = str(output_path.resolve().absolute())
                    this_exp_metadata[metadata_tag]['filename'] = str_path
                    frame_shape = splitter.frame_shape(
                                       i_roi=roi_index,
                                       z_value=z_value)
                    this_exp_metadata[metadata_tag]['height'] = frame_shape[0]
                    this_exp_metadata[metadata_tag]['width'] = frame_shape[1]

                    elapsed_time = time.time()-t0

                    self.logger.info("wrote "
                                     f"{output_path.resolve().absolute()} "
                                     f"after {elapsed_time:.2e} seconds")

                experiment_metadata.append(this_exp_metadata)

        # because the timeseries TIFFs are so big, we split
        # them differently to avoid reading through the TIFFs
        # more than once

        output_path_lookup = dict()
        for plane_group in self.args["plane_groups"]:
            for experiment in plane_group["ophys_experiments"]:
                exp_id = experiment["experiment_id"]
                scanfield_z = experiment["scanfield_z"]
                roi_index = experiment["roi_index"]
                experiment_dir = pathlib.Path(experiment["storage_directory"])
                fname = f"{exp_id}.h5"
                output_path = experiment_dir / fname
                output_path_lookup[(roi_index, scanfield_z)] = output_path

                frame_shape = timeseries_splitter.frame_shape(
                                       i_roi=roi_index,
                                       z_value=scanfield_z)

                str_path = str(output_path.resolve().absolute())

                for metadata in experiment_metadata:
                    if metadata["experiment_id"] == exp_id:
                        metadata["timeseries"]["height"] = frame_shape[0]
                        metadata["timeseries"]["width"] = frame_shape[1]
                        metadata["timeseries"]["filename"] = str_path

        timeseries_splitter.write_output_files(
                output_path_map=output_path_lookup,
                tmp_dir=self.args["tmp_dir"],
                dump_every=self.args["dump_every"],
                logger=self.logger)

        output["experiment_output"] = experiment_metadata

        ready_to_archive.add(self.args["surface_tif"])
        ready_to_archive.add(self.args["depths_tif"])
        ready_to_archive.add(self.args["timeseries_tif"])
        for zstack_path in zstack_path_list:
            ready_to_archive.add(str(zstack_path.resolve().absolute()))

        output["ready_to_archive"] = list(ready_to_archive)

        full_field_path = self.get_full_field_path()

        if full_field_path is not None:
            avg_path = self.args["surface_tif"]
            output_dir = pathlib.Path(self.args["storage_directory"])

            session_id = self.args["session_id"]

            # get session_id from the name of the directory where
            # the output files are being written
            if session_id is None:
                session_id = output_dir.name.split('_')[-1]

            output_name = f"{session_id}_stitched_full_field_img.h5"
            output_path = output_dir / output_name
            self.logger.info(f"Writing {output_path.resolve().absolute()}")

            write_out_stitched_full_field_image(
                path_to_avg_tiff=pathlib.Path(avg_path),
                path_to_full_field_tiff=full_field_path,
                output_path=output_path)

            self.logger.info("Wrote full field stitched image to "
                             f"{output_path.resolve().absolute()}")

        # record file metadata
        file_metadata = []
        for file_path in files_to_record:
            tiff_metadata = ScanImageMetadata(file_path)
            this_metadata = dict()
            this_metadata['input_tif'] = str(file_path.resolve().absolute())
            this_metadata['scanimage_metadata'] = tiff_metadata._metadata[0]
            this_metadata['roi_metadata'] = tiff_metadata._metadata[1]
            file_metadata.append(this_metadata)
        output["file_metadata"] = file_metadata

        self.output(get_sanitized_json_data(output), indent=1)
        duration = time.time()-t0
        self.logger.info(f"that took {duration:.2e} seconds")

    def get_full_field_path(self) -> Optional[pathlib.Path]:
        """
        Get the path to the full field image, if it exists.
        Return as a pathlib.Path.
        If the image does not exist, log the reason and return None.
        """
        stitch_full_field = True
        platform_key = "platform_json_path"
        if platform_key not in self.args or self.args[platform_key] is None:
            self.logger.warning(
                "platform_json_path not specified; "
                "skipping stitched full field image generation")
            stitch_full_field = False

        dir_key = "data_upload_dir"
        if dir_key not in self.args or self.args[dir_key] is None:
            self.logger.warning(
                "data_upload_dir not specified; "
                "skipping stitched full field image generation")
            stitch_full_field = False

        if not stitch_full_field:
            return None

        with open(self.args[platform_key], "rb") as in_file:
            platform_json_data = json.load(in_file)

        ff_key = "fullfield_2p_image"
        if ff_key not in platform_json_data:
            self.logger.warning(
                f"{ff_key} not present in "
                f"{self.args[platform_key]}; "
                "skipping stitched full field image generation")

            return None

        upload_dir = pathlib.Path(self.args[dir_key])
        full_field_path = upload_dir / platform_json_data[ff_key]
        if not full_field_path.is_file():
            self.logger.warning(f"{full_field_path.resolve().absolute()} "
                                "does not exist; skipping full field image "
                                "generation")
            return None

        return full_field_path


if __name__ == "__main__":
    runner = TiffSplitterCLI()
    runner.run()
