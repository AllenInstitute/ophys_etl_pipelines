import os
import argparse
import json
import h5py
from tifffile import imsave
from mesoscope_2p.metadata import SI_stringify_floats
from mesoscope_2p.tiff import MesoscopeTiff
from mesoscope_2p.conversion_utils import dump_dict_as_attrs, scanfield_to_h5


def convert_to_h5(tiff_file, h5_file, scanfield_slice):
    meso_tiff = MesoscopeTiff(tiff_file)
    h5p = h5py.File(h5_file, "w")
    meta_group = dump_dict_as_attrs(h5p, "header_data",
                                    meso_tiff.frame_metadata)
    dump_dict_as_attrs(meta_group, None, meso_tiff.roi_metadata)
    for z, scanfield in meso_tiff.scanfields.items():
        name = "scanfield_{}".format(z)
        roi_view = meso_tiff.roi_view(z)
        scanfield_to_h5(h5p, name, roi_view[scanfield_slice], scanfield)


def convert_to_tiffs(tiff_file, output_folder, scanfield_slice):
    meso_tiff = MesoscopeTiff(tiff_file)
    header_data = meso_tiff.frame_metadata.copy()
    header_data.update(meso_tiff.roi_metadata)
    meta_file = os.path.join(output_folder, "metadata.json")
    if os.path.exists(meta_file):
        raise RuntimeError("Output file {} already exists".format(meta_file))
    with open(meta_file, "w") as f:
        json.dump(SI_stringify_floats(
            header_data), f, indent=1, allow_nan=False)
    for z, scanfield in meso_tiff.scanfields.items():
        fname = os.path.join(output_folder, "scanfield_{}.tif".format(z))
        if os.path.exists(fname):
            raise RuntimeError("Output file {} already exists".format(fname))
        imsave(fname, meso_tiff.roi_view(z)[scanfield_slice], bigtiff=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Source tiff file from the mesoscope")
    parser.add_argument(
        "--output_path", type=str, required=True,
        help=("Destination path, should be a filename for hdf5 or a folder for"
              "tiffs"))
    parser.add_argument(
        "--frame_stop", type=int, default=None,
        help="Frame of the scanfield to stop at")
    parser.add_argument(
        "--frame_step", type=int, default=None,
        help="Get only every `frame_step` frames from each scanfield")
    parser.add_argument("--as_tiff", action="store_true")

    args = parser.parse_args()
    slc = slice(None, args.frame_stop, args.frame_step)
    if args.as_tiff:
        output_folder = args.output_path
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        convert_to_tiffs(args.input_file, output_folder, slc)
    else:
        h5_file = args.output_path
        if os.path.exists(h5_file):
            raise RuntimeError("Output file {} already exists".format(h5_file))
        convert_to_h5(args.input_file, h5_file, slc)


if __name__ == "__main__":
    main()