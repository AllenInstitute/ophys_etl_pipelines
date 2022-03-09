from argschema import ArgSchemaParser
import pathlib
import time

from ophys_etl.modules.mesoscope_splitting.schemas import (
    InputSchema, OutputSchema)

from ophys_etl.modules.mesoscope_splitting.tiff_splitter import (
    ScanImageTiffSplitter,
    TimeSeriesSplitter)

from ophys_etl.modules.mesoscope_splitting.zstack_splitter import (
    ZStackSplitter)

from ophys_etl.modules.mesoscope_splitting.tiff_metadata import (
    ScanImageMetadata)


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
        depth_splitter = ScanImageTiffSplitter(tiff_path=depth_path)
        files_to_record.append(depth_path)

        surface_path = pathlib.Path(self.args["surface_tif"])
        surface_splitter = ScanImageTiffSplitter(tiff_path=surface_path)
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
                self.logger.warn(msg)

        # how far apart are we going to allow two ROI center
        # to be and still be considered "the same"
        center_dsq_tol = 1.0e-6

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

                # TODO make sure we record this metadata
                # for each file in the output json
                # https://github.com/AllenInstitute/ophys_etl_pipelines/blob/main/src/ophys_etl/modules/mesoscope_splitting/__main__.py#L47-L63

                experiment_dir = pathlib.Path(experiment["storage_directory"])
                experiment_id = experiment["experiment_id"]
                roi_index = experiment["roi_index"]
                scanfield_z = experiment["scanfield_z"]

                roi_center = depth_splitter.roi_center(i_roi=roi_index)
                depth_name = f"{experiment_id}_depth.tif"
                depth_out_path = experiment_dir / depth_name
                depth_splitter.write_output_file(
                                    i_roi=roi_index,
                                    z_value=scanfield_z,
                                    output_path=depth_out_path)
                str_path = str(depth_out_path.resolve().absolute())
                this_exp_metadata['depth_2p']['filename'] = str_path
                frame_shape = depth_splitter.frame_shape(
                                       i_roi=roi_index,
                                       z_value=scanfield_z)
                this_exp_metadata['depth_2p']['height'] = frame_shape[0]
                this_exp_metadata['depth_2p']['width'] = frame_shape[1]

                self.logger.info("wrote "
                                 f"{depth_out_path.resolve().absolute()}")

                surface_name = f"{experiment_id}_surface.tif"
                surface_out_path = experiment_dir / surface_name
                surface_center = surface_splitter.roi_center(i_roi=roi_index)
                str_path = str(surface_out_path.resolve().absolute())
                this_exp_metadata['surface_2p']['filename'] = str_path
                frame_shape = surface_splitter.frame_shape(
                                         i_roi=roi_index,
                                         z_value=None)
                this_exp_metadata['surface_2p']['height'] = frame_shape[0]
                this_exp_metadata['surface_2p']['width'] = frame_shape[1]

                # check that the depth and surface ROIs are aligned
                center_dsq = ((surface_center[0]-roi_center[0])**2
                              + (surface_center[1]-roi_center[1])**2)

                if center_dsq > center_dsq_tol:
                    msg = f"experiment {experiment_id}\n"
                    msg += f"depth roi center {roi_center}\n"
                    msg += f"surface roi center {surface_center}\n"
                    msg += "are inconsistent"
                    raise RuntimeError(msg)

                surface_splitter.write_output_file(
                                    i_roi=roi_index,
                                    z_value=None,
                                    output_path=surface_out_path)

                self.logger.info(
                      f"wrote {surface_out_path.resolve().absolute()}")

                zstack_name = f"{experiment_id}_z_stack_local.h5"
                zstack_out_path = experiment_dir / zstack_name
                zstack_center = zstack_splitter.roi_center(i_roi=roi_index)
                str_path = str(zstack_out_path.resolve().absolute())
                this_exp_metadata['local_z_stack']['filename'] = str_path

                # check that the depth and zstack ROIs are aligned
                center_dsq = ((zstack_center[0]-roi_center[0])**2
                              + (zstack_center[1]-roi_center[1])**2)

                if center_dsq > center_dsq_tol:
                    msg = f"experiment {experiment_id}\n"
                    msg += f"depth roi center {roi_center}\n"
                    msg += f"zstack roi center {zstack_center}\n"
                    msg += "are inconsistent"
                    raise RuntimeError(msg)

                zstack_splitter.write_output_file(
                                    i_roi=roi_index,
                                    z_value=scanfield_z,
                                    output_path=zstack_out_path)

                frame_shape = zstack_splitter.frame_shape(
                                    i_roi=roi_index,
                                    z_value=scanfield_z)

                this_exp_metadata['local_z_stack']['height'] = frame_shape[0]
                this_exp_metadata['local_z_stack']['width'] = frame_shape[1]

                self.logger.info(
                      f"wrote {zstack_out_path.resolve().absolute()}")

                timeseries_name = f"{experiment_id}.h5"
                timeseries_out_path = experiment_dir / timeseries_name
                timeseries_center = timeseries_splitter.roi_center(
                                                            i_roi=roi_index)
                str_path = str(timeseries_out_path.resolve().absolute())
                this_exp_metadata['timeseries']['filename'] = str_path

                # check that the depth and timeseries ROIs are aligned
                center_dsq = ((timeseries_center[0]-roi_center[0])**2
                              + (timeseries_center[1]-roi_center[1])**2)

                if center_dsq > center_dsq_tol:
                    msg = f"experiment {experiment_id}\n"
                    msg += f"depth roi center {roi_center}\n"
                    msg += f"surface roi center {timeseries_center}\n"
                    msg += "are inconsistent"
                    raise RuntimeError(msg)

                timeseries_splitter.write_output_file(
                                        i_roi=roi_index,
                                        z_value=scanfield_z,
                                        output_path=timeseries_out_path)

                frame_shape = timeseries_splitter.frame_shape(
                                        i_roi=roi_index,
                                        z_value=scanfield_z)
                this_exp_metadata['timeseries']['height'] = frame_shape[0]
                this_exp_metadata['timeseries']['width'] = frame_shape[1]

                self.logger.info(
                       f"wrote {timeseries_out_path.resolve().absolute()}")

                experiment_metadata.append(this_exp_metadata)

        output["experiment_output"] = experiment_metadata

        ready_to_archive.add(self.args["surface_tif"])
        ready_to_archive.add(self.args["depths_tif"])
        ready_to_archive.add(self.args["timeseries_tif"])
        for zstack_path in zstack_path_list:
            ready_to_archive.add(str(zstack_path.resolve().absolute()))

        output["ready_to_archive"] = list(ready_to_archive)

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

        self.output(output, indent=1)
        duration = time.time()-t0
        self.logger.info(f"that took {duration:.2e} seconds")


if __name__ == "__main__":
    runner = TiffSplitterCLI()
    runner.run()
