import argschema
import tempfile
import os
import json
import ophys_etl.decrosstalk.decrosstalk_schema as decrosstalk_schema

from .utils import get_data_dir


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
        output = {}
        output['ophys_session_id'] = 9
        coupled_planes = []
        for i_pair in range(2):
            planes = []
            for ii in range(2):
                p = {}
                p['ophys_experiment_id'] = ii
                p['output_roi_trace_file'] = 'path/to/file'
                p['output_neuropil_trace_file'] = 'path/to/file'
                p['decrosstalk_invalid_raw'] = [1, 2, 3]
                p['decrosstalk_invalid_raw_active'] = [0, 5]
                p['decrosstalk_invalid_unmixed'] = [7, 8, 9]
                p['decrosstalk_invalid_unmixed_active'] = [10, 11]
                p['decrosstalk_ghost'] = [13, 14]
                planes.append(p)
            pair = {}
            pair['ophys_imaging_plane_group_id'] = 5
            pair['group_order'] = 0
            pair['planes'] = planes
            coupled_planes.append(pair)
        output['coupled_planes'] = coupled_planes

        self.output(output)


def get_schema_data():
    data_dir = get_data_dir()
    schema_fname = os.path.join(data_dir,
                                'ophys_plane_instantiation_data.json')

    with open(schema_fname, 'rb') as in_file:
        example_data = json.load(in_file)
    return example_data


def test_roi_schema():
    schema_data = get_schema_data()
    roi = schema_data['coupled_planes'][0]['planes'][0]['rois'][0]
    _ = DummyROILoader(input_data=roi, args=[])


def test_plane_schema():
    schema_data = get_schema_data()
    plane = schema_data['coupled_planes'][0]['planes'][0]
    _ = DummyPlaneLoader(input_data=plane, args=[])


def test_plane_pair_schema():
    schema_data = get_schema_data()
    plane_pair = schema_data['coupled_planes'][0]
    _ = DummyPlanePairLoader(input_data=plane_pair, args=[])


def test_decrosstalk_schema():
    schema_data = get_schema_data()
    _ = DummyDecrosstalkLoader(input_data=schema_data, args=[])


def test_output_schema(tmpdir):
    test_output_schema._temp_files = []
    tmp_fname = tempfile.mkstemp(prefix='output_schema_test_',
                                 suffix='.json',
                                 dir=tmpdir)[1]
    test_output_schema._temp_files.append(tmp_fname)

    dummy = DummySchemaOutput(args=['--output_json', '%s' % tmp_fname])
    dummy.run()
