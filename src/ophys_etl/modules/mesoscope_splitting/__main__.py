import logging
import os
import numpy as np
import h5py
from typing import (
    List, Dict, Tuple)
from typing_extensions import TypedDict
from argschema import ArgSchemaParser
from ophys_etl.modules.mesoscope_splitting.tiff import MesoscopeTiff
from ophys_etl.modules.mesoscope_splitting.conversion_utils import (
    volume_to_h5, volume_to_tif, average_and_unsign)
from ophys_etl.modules.mesoscope_splitting.metadata import SI_stringify_floats
from ophys_etl.modules.mesoscope_splitting.schemas import (
    InputSchema, OutputSchema)


def mock_h5(*args, **kwargs):
    pass


def mock_tif(filename, *args, **kwargs):
    f = open(filename, "w")
    f.close()


class ConversionOutputDict(TypedDict):
    filename: str
    resolution: float
    offset_x: float
    offset_y: float
    rotation: float
    height: int
    width: int


class ConversionMetadataDict(TypedDict):
    input_tif: str
    roi_metadata: Dict
    scanimage_metadata: Dict


ConversionTuple = Tuple[ConversionOutputDict, ConversionMetadataDict]


def conversion_output(volume,
                      outfile,
                      experiment_info) -> ConversionTuple:
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


def convert_column(input_tif, session_storage, experiment_info):
    mt = MesoscopeTiff(input_tif)
    if len(mt.volume_views) != 1:
        raise ValueError("Expected 1 stack in {}, but found {}".format(
            input_tif, len(mt.volume_views)))
    basename = os.path.basename(input_tif)
    h5_base = os.path.splitext(basename)[0] + ".h5"
    filename = os.path.join(session_storage, h5_base)
    stack = mt.volume_views[0]

    with h5py.File(filename, "w") as f:
        volume_to_h5(f, stack)

    return conversion_output(mt.volume_views[0], filename, experiment_info)


def split_z(input_tif: MesoscopeTiff,
            experiments: List[Dict],
            testing=False) -> ConversionTuple:
    """Takes a z_stack file from a Mesoscope ophys session and a list
    of the experiments performed during that session and splits the data
    into a z_stack file (as a .h5) for each individual experiment.

    Parameters
    ----------
    input_tif : MesoscopeTiff
        MesoscopeTiff object containing the timeseries data that
        will be split by experiment.

    experiments : List[Dict]
        A list of dictionaries, each containing information about
        the experiments performed during the ophys session, as described by
        the ExperimentPlane schema defined in /pipelines/brain_observatory/
        schemas/split_mesoscope.py.

    Returns
    -------
    outs : ConversionOutputDict
        A dictionary containing specific information about the experiments,
        including the location of the input data.

    meta : ConversionMetadataDict
        A dictionary containing metadata about the experiment, including the
        location where the ata will be saved.
    """
    directory = experiments["storage_directory"]
    eid = experiments["experiment_id"]
    filename = os.path.join(directory, "{}_z_stack_local.h5".format(eid))

    i = experiments["roi_index"]
    z = experiments["scanfield_z"]
    stack = input_tif.nearest_volume(i, z)
    if stack is None:
        raise ValueError(
            "Could not find stack to extract from {} for experiment {}".format(
                input_tif._source, eid
            )
        )

    logging.info(
        "Got stack centered at z={} for target z={} in {}".format(
            np.mean(stack.zs), z, input_tif._source
        )
    )

    with h5py.File(filename, "w") as f:
        volume_to_h5(f, stack)

    outs, meta = conversion_output(stack, filename, experiments)

    return outs, meta


def split_image(input_tif: MesoscopeTiff,
                experiments: List[Dict],
                name: str) -> Tuple[Dict, Dict]:
    """Takes a file from a Mesoscope ophys session containing many images
    and a list of the experiments performed during that session and splits
    the images into multiple .tif files by experiment.

    Parameters
    ----------
    input_tif : MesoscopeTiff
        MesoscopeTiff object containing the timeseries data that
        will be split by experiment.

    experiments : List[Dict]
        A list of dictionaries, each containing information about
        the experiments performed during the ophys session, as described by
        the ExperimentPlane schema defined in /pipelines/brain_observatory/
        schemas/split_mesoscope.py.

    name : str
        The name of the data that will be split (e.g. 'surface'
        for surface image data or 'depth' for depth image data)

    Returns
    -------
    outs : ConversionOutputDict
        A dictionary containing specific information about the experiments,
        including the location of the input data.

    meta : ConversionMetadataDict
        A dictionary containing metadata about the experiment, including the
        location where the ata will be saved.
    """
    outs = {}

    for exp in experiments:
        directory = exp["storage_directory"]
        eid = exp["experiment_id"]
        i = exp["roi_index"]
        z = exp["scanfield_z"]
        filename = os.path.join(directory, "{}_{}.tif".format(eid, name))

        plane = input_tif.nearest_plane(i, z)
        if plane is None:
            raise ValueError(
                "No plane to extract from {} for experiment {}".format(
                    input_tif._source, eid
                )
            )

        logging.info(
            "Got plane at z={} for target z={} in {}".format(
                np.mean(plane.zs), z, input_tif._source
            )
        )

        volume_to_tif(filename, plane, projection_func=average_and_unsign)

        outs[eid], meta = conversion_output(plane, filename, exp)

    return outs, meta


def split_timeseries(input_tif: MesoscopeTiff,
                     experiments: List[Dict]) -> Tuple[Dict, Dict]:
    """Takes a timeseries file from a Mesoscope ophys session and a list
    of the experiments performed during that session and splits the data
    into a timeseries file (as a .h5) for each individual experiment.

    Parameters
    ----------
    input_tif : MesoscopeTiff
        MesoscopeTiff object containing the timeseries data that
        will be split by experiment.

    experiments : List[Dict]
        A list of dictionaries, each containing information about
        the experiments performed during the ophys session, as described by
        the ExperimentPlane schema defined in /pipelines/brain_observatory/
        schemas/split_mesoscope.py.

    Returns
    -------
    outs : ConversionOutputDict
        A dictionary containing specific information about the experiments,
        including the location of the input data.

    meta : ConversionMetadataDict
        A dictionary containing metadata about the experiment, including the
        location where the ata will be saved.
    """
    outs = {}

    for exp in experiments:
        directory = exp["storage_directory"]
        eid = exp["experiment_id"]
        i = exp["roi_index"]
        z = exp["scanfield_z"]
        filename = os.path.join(directory, "{}.h5".format(eid))

        plane = input_tif.nearest_plane(i, z)
        if plane is None:
            raise ValueError(
                "No plane to extract from {} for experiment {}".format(
                    input_tif._source, eid
                )
            )

        logging.info(
            "Got plane at z={} for target z={} in {}".format(
                np.mean(plane.zs), z, input_tif._source
            )
        )

        with h5py.File(filename, "w") as f:
            volume_to_h5(f, plane)

        outs[eid], meta = conversion_output(plane, filename, exp)
        if input_tif.is_multiscope:
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

    ts_mesoscope_tiff = MesoscopeTiff(mod.args["timeseries_tif"])
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
    main()
