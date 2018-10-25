import logging
import os
import numpy as np
import h5py
from argschema import ArgSchemaParser
from mesoscope_2p import MesoscopeTiff
from mesoscope_2p.conversion_utils import (volume_to_h5, volume_to_tif, 
                                           average_and_unsign)
from mesoscope_2p.metadata import SI_stringify_floats
from brain_observatory_utils.schemas.split_mesoscope import (InputSchema,
                                                             OutputSchema)


def mock_h5(*args, **kwargs):
    pass


def mock_tif(filename, *args, **kwargs):
    f = open(filename, "w")
    f.close()


def conversion_output(volume, outfile, experiment_info):
    mt = volume._tiff
    height, width = volume.plane_shape
    meta_res = {"input_tif": mt._source,
                "roi_metadata": SI_stringify_floats(mt.roi_metadata),
                "scanimage_metadata": SI_stringify_floats(mt.frame_metadata)}
    out_res = {"filename": outfile,
               "resolution": experiment_info["resolution"],
               "offset_x": experiment_info["offset_x"],
               "offset_y": experiment_info["offset_y"],
               "rotation": experiment_info["rotation"],
               "height": height,
               "width": width}

    return out_res, meta_res


def convert_column(input_tif, session_storage, experiment_info, **h5_opts):
    mt = MesoscopeTiff(input_tif)
    if len(mt.volume_views) != 1:
        raise ValueError("Expected 1 stack in {}, but found {}".format(
            input_tif, len(mt.volume_views)))
    basename = os.path.basename(input_tif)
    h5_base = os.path.splitext(basename)[0] + ".h5"
    filename = os.path.join(session_storage, h5_base)
    with h5py.File(filename, "w") as f:
        volume_to_h5(f, mt.volume_views[0], **h5_opts)

    return conversion_output(mt.volume_views[0], filename, experiment_info)


def split_z(input_tif, experiment_info, **h5_opts):
    directory = experiment_info["storage_directory"]
    eid = experiment_info["experiment_id"]
    filename = os.path.join(directory, "{}_z_stack_local.h5".format(eid))

    mt = MesoscopeTiff(input_tif)

    i = experiment_info["roi_index"]
    z = experiment_info["scanfield_z"]
    stack = mt.nearest_volume(i, z)
    if stack is None:
        raise ValueError(
            "Could not find stack to extract from {} for experiment {}".format(
                input_tif, eid
            )
        )

    logging.info(
        "Got stack centered at z={} for target z={} in {}".format(
            np.mean(stack.zs), z, input_tif
        )
    )

    with h5py.File(filename, "w") as f:
        volume_to_h5(f, stack, **h5_opts)

    return conversion_output(stack, filename, experiment_info)


def split_image(input_tif, experiments, name):
    mt = MesoscopeTiff(input_tif)
    outs = {}

    for exp in experiments:
        directory = exp["storage_directory"]
        eid = exp["experiment_id"]
        i = exp["roi_index"]
        z = exp["scanfield_z"]
        filename = os.path.join(directory, "{}_{}.tif".format(eid, name))

        plane = mt.nearest_plane(i, z)
        if plane is None:
            raise ValueError(
                "No plane to extract from {} for experiment {}".format(
                    input_tif, eid
                )
            )

        logging.info(
            "Got plane at z={} for target z={} in {}".format(
                np.mean(plane.zs), z, input_tif
            )
        )

        volume_to_tif(filename, plane, projection_func=average_and_unsign)

        outs[eid], meta = conversion_output(plane, filename, exp)

    return outs, meta


def split_timeseries(input_tif, experiments, **h5_opts):
    mt = MesoscopeTiff(input_tif)
    outs = {}

    for exp in experiments:
        directory = exp["storage_directory"]
        eid = exp["experiment_id"]
        i = exp["roi_index"]
        z = exp["scanfield_z"]
        filename = os.path.join(directory, "{}.h5".format(eid))

        plane = mt.nearest_plane(i, z)
        if plane is None:
            raise ValueError(
                "No plane to extract from {} for experiment {}".format(
                    input_tif, eid
                )
            )

        logging.info(
            "Got plane at z={} for target z={} in {}".format(
                np.mean(plane.zs), z, input_tif
            )
        )

        with h5py.File(filename, "w") as f:
            volume_to_h5(f, plane, **h5_opts)

        outs[eid], meta = conversion_output(plane, filename, exp)
        if mt.is_multiscope:
            outs[eid]["sync_stride"] = plane.stride // 2
            outs[eid]["sync_offset"] = plane.page_offset // 2
        else:
            outs[eid]["sync_stride"] = plane.stride
            outs[eid]["sync_offset"] = plane.page_offset

    return outs, meta


def main():
    mod = ArgSchemaParser(schema_type=InputSchema,
                          output_schema_type=OutputSchema)

    if mod.args['test_mode']:
        global volume_to_h5, volume_to_tif
        volume_to_h5 = mock_h5
        volume_to_tif = mock_tif

    h5_opts = {}
    if mod.args['compression_level']:
        h5_opts = {"compression": "gzip",
                   "compression_opts": mod.args['compression_level']}

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
                        plane_group["ophys_experiments"][0],
                        **h5_opts
                    )
                    output["column_stacks"].append(out)
                    output["file_metadata"].append(meta)
                except ValueError as e:
                    logging.error(e) # don't break on failed column stack conversion
                stack_tifs.add(column_stack)
        for exp in plane_group["ophys_experiments"]:
            localz = plane_group["local_z_stack_tif"]
            ready_to_archive.add(localz)
            out, meta = split_z(localz, exp, **h5_opts)
            if localz not in stack_tifs:
                output["file_metadata"].append(meta)
                stack_tifs.add(localz)
            experiments.append(exp)
            z_outs[exp["experiment_id"]] = out

    surf_outs, surf_meta = split_image(mod.args["surface_tif"],
                                       experiments,
                                       "surface")
    depth_outs, depth_meta = split_image(mod.args["depths_tif"],
                                         experiments,
                                         "depth")
    ts_outs, ts_meta = split_timeseries(mod.args["timeseries_tif"],
                                        experiments,
                                        **h5_opts)

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
    main()
