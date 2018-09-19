import os
import argparse
import json
import h5py
from mesoscope_2p.metadata import SI_stringify_floats
from mesoscope_2p.tiff import MesoscopeTiff
from mesoscope_2p.conversion_utils import (volume_to_h5, volume_to_tif,
                                           average_and_unsign)


def get_outfile(plane, folder, prefix=None, extension="tif"):
    if plane.metadata["name"]:
        name = plane.metadata["name"]
    else:
        name = "scanfield"
    fname = "{}_{}.{}".format(name, plane.z, extension)
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


def convert_to_h5(tiff_file, output_folder, scanfield_slice, clobber=False,
                  prefix=None, **h5_opts):
    meso_tiff = MesoscopeTiff(tiff_file)
    filename = os.path.join(output_folder, "{}_metadata.json".format(os.path.basename(tiff_file)))
    dump_metadata(filename, meso_tiff, clobber=clobber)
    for plane in meso_tiff.planes:
        filename = get_outfile(plane, output_folder, prefix, "h5")
        if os.path.exists(filename) and not clobber:
            raise RuntimeError("Output file {} already exists".format(filename))
        with h5py.File(filename, "w") as f:
            volume_to_h5(f, plane[scanfield_slice], **h5_opts)


def convert_to_tiffs(tiff_file, output_folder, scanfield_slice, clobber=False,
                     prefix=None, projection_func=None):
    meso_tiff = MesoscopeTiff(tiff_file)
    filename = os.path.join(output_folder, "{}_metadata.json".format(os.path.basename(tiff_file)))
    dump_metadata(filename, meso_tiff, clobber=clobber)
    for plane in meso_tiff.planes:
        filename = get_outfile(plane, output_folder, prefix, "tif")
        if os.path.exists(filename) and not clobber:
            raise RuntimeError("Output file {} already exists".format(filename))
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
        "--compression_level", type=int, default=None,
        help=("Gzip compression level for hdf5, defaults to no compression. "
              "Should be 1-9."))
    parser.add_argument(
        "--averaged", action="store_true",
        help="Flag to compute averaged tif, if tif output.")

    args = parser.parse_args()
    slc = slice(None, args.frame_stop, args.frame_step)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if args.as_tiff:
        if args.averaged:
            projection_func = average_and_unsign
        else:
            projection_func = None
        convert_to_tiffs(args.input_file, args.output_path, slc, args.clobber,
                         args.prefix, projection_func)
    else:
        convert_to_h5(args.input_file, args.output_path, slc, args.clobber,
                      args.prefix)


if __name__ == "__main__":
    main()