import os
import argparse
import h5py
from mesoscope_2p.tiff import MesoscopeTiff
from mesoscope_2p.conversion_utils import dump_dict_as_attrs, scanfield_to_h5


def convert(tiff_file, h5_file, scanfield_slice):
    meso_tiff = MesoscopeTiff(tiff_file)
    h5p = h5py.File(h5_file, "w")
    meta_group = dump_dict_as_attrs(h5p, "header_data",
                                    meso_tiff.frame_metadata)
    dump_dict_as_attrs(meta_group, None, meso_tiff.roi_metadata)
    for z, scanfield in meso_tiff.scanfields.items():
        name = "scanfield_{}".format(z)
        roi_view = meso_tiff.roi_view(z)
        scanfield_to_h5(h5p, name, roi_view[scanfield_slice], scanfield)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, required=True,
        help="Source tiff file from the mesoscope")
    parser.add_argument(
        "--output_file", type=str, required=True,
        help="Destination hdf5 filename")
    parser.add_argument(
        "--frame_stop", type=int, default=None,
        help="Frame of the scanfield to stop at")
    parser.add_argument(
        "--frame_step", type=int, default=None,
        help="Get only every `frame_step` frames from each scanfield")

    args = parser.parse_args()
    slc = slice(None, args.frame_stop, args.frame_step)
    h5_file = args.output_file
    if os.path.exists(h5_file):
        raise RuntimeError("Output file {} already exists".format(h5_file))
    convert(args.input_file, h5_file, slc)


if __name__ == "__main__":
    main()