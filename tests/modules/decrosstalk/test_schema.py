import argschema
import tempfile
import os
import copy
import ophys_etl.modules.decrosstalk.decrosstalk_schema as decrosstalk_schema


class DummyDecrosstalkLoader(argschema.ArgSchemaParser):
    default_schema = decrosstalk_schema.DecrosstalkInputSchema


class DummySchemaOutput(argschema.ArgSchemaParser):
    default_schema = decrosstalk_schema.DecrosstalkInputSchema
    default_output_schema = decrosstalk_schema.DecrosstalkOutputSchema

    def run(self):
        eg_plane = self.args['coupled_planes'][0]['planes'][0]
        output = {}
        output['ophys_session_id'] = 9
        coupled_planes = []
        for i_pair in range(2):
            planes = []
            for ii in range(2):
                p = {}
                p['ophys_experiment_id'] = ii
                fname = eg_plane['output_roi_trace_file']
                p['output_roi_trace_file'] = fname
                fname = eg_plane['output_neuropil_trace_file']
                p['output_neuropil_trace_file'] = fname
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


def get_schema_data(tmpdir: str,
                    example_data: dict):
    """
    Update a dict representation of our schema for unit testing purposes

    Parameters
    ----------
    tmpdir: str
        directory where test data products can be written
    example_data: dict
        dict representation of the Decrosstalk schema

    Returns
    -------
    example_data updated with temporary output paths for testing
    """
    example_data = copy.deepcopy(example_data)
    # Because the input schema is going to verify that the
    # motion_corrected_stack and maximum_projection_image_file
    # actually exist, we must create it in tmp
    dummy_movie_fname = os.path.join(tmpdir, 'dummy_movie.h5')
    with open(dummy_movie_fname, 'w') as out_file:
        out_file.write('hi')

    dummy_max_fname = os.path.join(tmpdir, 'dummy_img.png')
    with open(dummy_max_fname, 'w') as out_file:
        out_file.write('hi')

    for pair in example_data['coupled_planes']:
        for plane in pair['planes']:
            plane['motion_corrected_stack'] = dummy_movie_fname
            plane['maximum_projection_image_file'] = dummy_max_fname
            fname = os.path.join(tmpdir, 'roi_file.h5')
            plane['output_roi_trace_file'] = fname
            fname = os.path.join(tmpdir, 'np_file.h5')
            plane['output_neuropil_trace_file'] = fname

    example_data['qc_output_dir'] = os.path.join(tmpdir, 'qc')
    return example_data


def test_decrosstalk_schema(tmpdir, ophys_plane_data_fixture):
    schema_data = get_schema_data(tmpdir,
                                  ophys_plane_data_fixture)
    _ = DummyDecrosstalkLoader(input_data=schema_data, args=[])


def test_output_schema(tmpdir,
                       ophys_plane_data_fixture):
    schema_data = get_schema_data(tmpdir,
                                  ophys_plane_data_fixture)
    test_output_schema._temp_files = []
    tmp_fname = tempfile.mkstemp(prefix='output_schema_test_',
                                 suffix='.json',
                                 dir=tmpdir)[1]
    test_output_schema._temp_files.append(tmp_fname)

    dummy = DummySchemaOutput(input_data=schema_data,
                              args=['--output_json', '%s' % tmp_fname])
    dummy.run()
