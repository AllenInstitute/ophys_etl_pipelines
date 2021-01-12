import argschema
import tempfile
import os
import json
import ophys_etl.decrosstalk.decrosstalk_schema as decrosstalk_schema

class DummyDecrosstalkLoader(argschema.ArgSchemaParser):
    default_schema = decrosstalk_schema.DecrosstalkInputSchema

class DummyROILoader(argschema.ArgSchemaParser):
    default_schema = decrosstalk_schema.RoiSchema

class DummyPlaneLoader(argschema.ArgSchemaParser):
    default_schema = decrosstalk_schema.PlaneSchema

class DummyPlanePairLoader(argschema.ArgSchemaParser):
    default_schema = decrosstalk_schema.PlanePairSchema

class DummySchemaOutput(argschema.ArgSchemaParser):
    default_output_schema = decrosstalk_schema.DecrosstalkOutputSchema

    def run(self):
        output = {'decrosstalk_invalid_raw_trace':[],
                  'decrosstalk_invalid_unmixed_trace':[1,2,3,4],
                  'decrosstalk_ghost_roi_ids':[]}
        self.output(output)

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

def test_output_schema():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    tmp_dir = os.path.join(this_dir, 'tmp')
    assert os.path.isdir(tmp_dir)
    tmp_fname = tempfile.mkstemp(prefix='output_schema_test_',
                                 suffix='.json',
                                 dir=tmp_dir)[1]
    try:
        dummy = DummySchemaOutput(args=['--output_json', '%s' % tmp_fname])
        dummy.run()
    except:
        raise
    finally:
        if os.path.exists(tmp_fname):
            os.unlink(tmp_fname)
