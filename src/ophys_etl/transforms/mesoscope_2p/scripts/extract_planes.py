import os
import argparse
import json
import numpy as np

from hashlib import sha256
from ophys_etl.transforms.mesoscope_2p.metadata import SI_stringify_floats
from ophys_etl.transforms.mesoscope_2p.tiff import MesoscopeTiff
from ophys_etl.transforms.mesoscope_2p.conversion_utils import (
    volume_to_tif, average_and_unsign)


def get_outfile(plane, folder, prefix=None, extension="tif"):
    if plane.flagged:
        name = "no_roi_match" + sha256(plane.asarray()).hexdigest()
    elif plane.metadata["name"]:
        name = plane.metadata["name"]
    else:
        name = "scanfield"
    z = int(np.mean(plane.zs))
    fname = "{}_{}.{}".format(name, z, extension)
    if prefix:
        fname = "{}_{}".format(prefix, fname)

    return os.path.join(folder, fname)


def dump_metadata(filename, meso_tiff, clobber=False):
    if os.path.exists(filename) and not clobber:
        raise RuntimeError("Output file {} already exists".format(filename))
    header_data = meso_tiff.frame_metadata.copy()
    header_data.update(meso_tiff.roi_metadata)
    with open(filename, "w") as f:
        json.dump(SI_stringify_floats(
            header_data), f, indent=1, allow_nan=False)


def convert_to_tiffs(tiff_file, output_folder, scanfield_slice, clobber=False,
                     prefix=None, projection_func=None,
                     view_attr="plane_views"):
    meso_tiff = MesoscopeTiff(tiff_file)
    filename = os.path.join(
        output_folder, "{}_metadata.json".format(os.path.basename(tiff_file)))
    dump_metadata(filename, meso_tiff, clobber=clobber)
    for plane in getattr(meso_tiff, view_attr):
        filename = get_outfile(plane, output_folder, prefix, "tif")
        if os.path.exists(filename) and not clobber:
            raise RuntimeError(
                "Output file {} already exists".format(filename))
        volume_to_tif(filename, plane[scanfield_slice], projection_func)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Source tiff file from the mesoscope.")
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Destination folder that will contain output files.")
    parser.add_argument(
        "--frame_stop", type=int, default=None,
        help="Frame of the scanfield to stop at.")
    parser.add_argument(
        "--frame_step", type=int, default=None,
        help="Get only every `frame_step` frames from each scanfield.")
    parser.add_argument(
        "--as_tiff", action="store_true",
        help="Flag to store output as tiff files.")
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Prefix to prepend to output filenames")
    parser.add_argument(
        "--clobber", action="store_true",
        help=("Flag to clobber existing output files."))
    parser.add_argument(
        "--averaged", action="store_true",
        help="Flag to compute averaged tif, if tif output.")
    parser.add_argument(
        "--as_stack", action="store_true",
        help="Flag to extract as stacks.")

    args = parser.parse_args()
    slc = slice(None, args.frame_stop, args.frame_step)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.averaged:
        projection_func = average_and_unsign
    else:
        projection_func = None
    if args.as_stack:
        view_attr = "volume_views"
    else:
        view_attr = "plane_views"
    convert_to_tiffs(args.input_file, args.output_path, slc, args.clobber,
                     args.prefix, projection_func, view_attr)


if __name__ == "__main__":
    main()
