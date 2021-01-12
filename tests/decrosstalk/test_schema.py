import argschema
import os
import json
import ophys_etl.decrosstalk.decrosstalk_schema as decrosstalk_schema

class DummyDecrosstalkLoader(argschema.ArgSchemaParser):
    default_schema = decrosstalk_schema.DecrosstalkSchema

class DummyROILoader(argschema.ArgSchemaParser):
    default_schema = decrosstalk_schema.RoiSchema

class DummyPlaneLoader(argschema.ArgSchemaParser):
    default_schema = decrosstalk_schema.PlaneSchema

class DummyPlanePairLoader(argschema.ArgSchemaParser):
    default_schema = decrosstalk_schema.PlanePairSchema

def get_schema_data():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(this_dir, 'data')
    assert os.path.isdir(data_dir)
    schema_fname = os.path.join(data_dir,
                                'ophys_plane_instantiation_data.json')

    with open(schema_fname, 'rb') as in_file:
        example_data = json.load(in_file)
    return example_data

def test_roi_schema():
    schema_data = get_schema_data()
    roi = schema_data['coupled_planes'][0]['planes'][0]['rois'][0]
    dummy = DummyROILoader(input_data=roi, args=[])

def test_plane_schema():
    schema_data = get_schema_data()
    plane = schema_data['coupled_planes'][0]['planes'][0]
    dummy = DummyPlaneLoader(input_data=plane, args=[])

def test_plane_pair_schema():
    schema_data = get_schema_data()
    plane_pair = schema_data['coupled_planes'][0]
    dummy = DummyPlanePairLoader(input_data=plane_pair, args=[])

def test_decrosstalk_schema():
    schema_data = get_schema_data()
    dummy = DummyDecrosstalkLoader(input_data=schema_data, args=[])
