import os
import json
from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.decrosstalk.ophys_plane import DecrosstalkingOphysPlane

from .utils import get_data_dir


def get_ophys_test_schema():
    """
    Load and return the schema dict we will use to test
    DecrosstalkingOphysPlane instantiation

    The dict is artificially constructed to have 2 pairs of planes,
    each plane containing 2 ROIs.
    """
    data_dir = get_data_dir()
    data_file_name = os.path.join(data_dir,
                                  'ophys_plane_instantiation_data.json')

    assert os.path.isfile(data_file_name)
    with open(data_file_name, 'rb') as in_file:
        data = json.load(in_file)
    return data


def test_roi_instantiation():
    schema_dict = get_ophys_test_schema()
    ct = 0
    for meta_pair in schema_dict['coupled_planes']:
        pair = meta_pair['planes']
        for plane in pair:
            for roi_args in plane['rois']:
                _ = OphysROI.from_schema_dict(roi_args)
                ct += 1
    assert ct == 8


def test_plane_instantiation():
    schema_dict = get_ophys_test_schema()
    ct = 0
    for pair in schema_dict['coupled_planes']:
        for plane_args in pair['planes']:
            _ = DecrosstalkingOphysPlane.from_schema_dict(plane_args)
            ct += 1
    assert ct == 4
