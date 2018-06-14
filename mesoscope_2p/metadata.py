import struct
import logging
import math
import json
from six import string_types
from tifffile import stripnull
from tifffile.tifffile import read_scanimage_metadata


BYTEORDER = b'II'
BIGTIFF = 43
SCANIMAGE_TIFF_VERSION = 3
SCANIMAGE_TIFF_MAGIC = 117637889


def floatify_SI_float_strings(json_data):
    if isinstance(json_data, dict):
        for key, value in json_data.items():
            json_data[key] = floatify_SI_float_strings(value)
    elif isinstance(json_data, list):
        json_data = [floatify_SI_float_strings(item) for item in json_data]
    elif isinstance(json_data, string_types):
        if json_data == "_NaN_":
            json_data = float("nan")
        elif json_data == "_Inf_":
            json_data = float("inf")
        elif json_data == "_-Inf_":
            json_data = float("-inf")
    return json_data


def SI_stringify_floats(data):
    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = SI_stringify_floats(value)
    elif isinstance(data, list):
        data = [SI_stringify_floats(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data):
            data = "_NaN_"
        elif math.isinf(data) and data > 0:
            data = "_Inf_"
        elif math.isinf(data) and data < 0:
            data = "_-Inf_"
    return data


def unflatten_dict(flat_dict, key_sep="."):
    """Convert a flat dictionary to a nested dictionary."""
    nested_dict = {}
    for key, value in flat_dict.items():
        split_keys = key.split(key_sep)
        current_dict = nested_dict
        for nest_key in split_keys[:-1]:
            if nest_key not in current_dict:
                current_dict[nest_key] = {}
            current_dict = current_dict[nest_key]
        current_dict[split_keys[-1]] = value
    return nested_dict


def _tiff_header_data_2017b_v1(fp):
    """Extract ScanImage header data from a tiff for ScanImage 2017b 1.

    The header 2017b v0 and 2017b v1 have different header formats. It
    seems that tifffile doesn't recognize 2017b1 as a scanimage tiff,
    and the frame header data is stored in a different format.

    http://scanimage.vidriotechnologies.com/display/SI2016/ScanImage+BigTiff+Specification
    """
    fp.seek(0)
    byteorder, tiff_version = struct.unpack('<2sH', fp.read(4))
    if byteorder != BYTEORDER or tiff_version != BIGTIFF:
        raise ValueError("File is not a BigTIFF")
    fp.seek(16)
    magic, version, frame_data_size, roi_data_size = struct.unpack(
        '<IIII', fp.read(16))
    if magic != SCANIMAGE_TIFF_MAGIC or version != SCANIMAGE_TIFF_VERSION:
        raise ValueError("File is not a ScanImage BigTIFF v3")
    
    frame_data = json.loads(stripnull(fp.read(frame_data_size)).decode('utf-8'))
    roi_data = json.loads(stripnull(fp.read(roi_data_size)).decode('utf-8'))

    return frame_data, roi_data


def tiff_header_data(filename):
    """Extract ScanImage header data from a tiff.
    
    http://scanimage.vidriotechnologies.com/display/SI2016/ScanImage+BigTiff+Specification
    """
    with open(filename, "rb") as f:
        try:
            frame_data, roi_data = read_scanimage_metadata(f)
            frame_data = unflatten_dict(frame_data)
            logging.debug("Loaded %s as 2017b v0", filename)
        except ValueError:
            frame_data, roi_data = _tiff_header_data_2017b_v1(f)
            logging.debug("Loaded %s as 2017b v1", filename)
    frame_data = floatify_SI_float_strings(frame_data)
    roi_data = floatify_SI_float_strings(roi_data)
    return frame_data, roi_data
