from argschmea import ArgschemaParser


from ophys_etl.modules.mesoscope_splitting.schemas import (
    InputSchema, OutputSchema)


class TiffSplitterCLI(ArgSchemaParser):
    default_schema = InputSchema
    default_output_schema = OutputSchema

    def run(self):

        # check for repeated z value
        ts_mesoscope_tiff = MesoscopeTiff(mod.args["timeseries_tif"])
        checks.check_for_repeated_planes(ts_mesoscope_tiff)

        # check consistency between input json and tiff headers
        check_list = []
        for plane_group in mod.args["plane_groups"]:
            pg_tiff = Path(plane_group["local_z_stack_tif"])
            roi_index = list({i["roi_index"]
                              for i in plane_group["ophys_experiments"]})
            check_list.append(
                checks.ConsistencyInput(tiff=pg_tiff, roi_index=roi_index))
        checks.splitting_consistency_check(check_list)

        # end checks

        if mod.args['test_mode']:
            global volume_to_h5, volume_to_tif
            volume_to_h5 = mock_h5
            volume_to_tif = mock_tif

        stack_tifs = set()
        ready_to_archive = set()
        session_storage = mod.args["storage_directory"]

        output = {"column_stacks": [],
                  "file_metadata": []}

        experiments = []
        z_outs = {}

        for plane_group in mod.args["plane_groups"]:
            column_stack = plane_group.get("column_z_stack_tif", None)
            if column_stack:
                ready_to_archive.add(column_stack)
                if column_stack not in stack_tifs:
                    try:
                        out, meta = convert_column(
                            column_stack,
                            session_storage,
                            plane_group["ophys_experiments"][0])
                        output["column_stacks"].append(out)
                        output["file_metadata"].append(meta)
                    except ValueError as e:
                        # don't break on failed column stack conversion
                        logging.error(e)
                    stack_tifs.add(column_stack)

            for exp in plane_group["ophys_experiments"]:
                localz = plane_group["local_z_stack_tif"]
                ready_to_archive.add(localz)

                localz_tiff = MesoscopeTiff(localz)
                out, meta = split_z(localz_tiff, exp)

                if localz not in stack_tifs:
                    output["file_metadata"].append(meta)
                    stack_tifs.add(localz)

                experiments.append(exp)
                z_outs[exp["experiment_id"]] = out

        sf_mesoscope_tiff = MesoscopeTiff(mod.args["surface_tif"])
        surf_outs, surf_meta = split_image(sf_mesoscope_tiff,
                                           experiments,
                                           "surface")

        dp_mesoscope_tiff = MesoscopeTiff(mod.args["depths_tif"])
        depth_outs, depth_meta = split_image(dp_mesoscope_tiff,
                                             experiments,
                                             "depth")

        ts_outs, ts_meta = split_timeseries(ts_mesoscope_tiff,
                                            experiments)

        output["file_metadata"].extend([surf_meta, depth_meta, ts_meta])

        exp_out = []
        for exp in experiments:
            eid = exp["experiment_id"]
            sync_stride = ts_outs[eid].pop("sync_stride")
            sync_offset = ts_outs[eid].pop("sync_offset")
            exp_data = {"experiment_id": eid,
                        "local_z_stack": z_outs[eid],
                        "surface_2p": surf_outs[eid],
                        "depth_2p": depth_outs[eid],
                        "timeseries": ts_outs[eid],
                        "sync_offset": sync_offset,
                        "sync_stride": sync_stride}
            exp_out.append(exp_data)

        output["experiment_output"] = exp_out

        ready_to_archive.add(mod.args["surface_tif"])
        ready_to_archive.add(mod.args["depths_tif"])
        ready_to_archive.add(mod.args["timeseries_tif"])

        output["ready_to_archive"] = list(ready_to_archive)

        mod.output(output, indent=1)


if __name__ == "__main__":
    runner = TiffSplitterCLI()
    runner.run()
