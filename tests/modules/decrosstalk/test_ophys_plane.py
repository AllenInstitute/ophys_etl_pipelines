from ophys_etl.modules.decrosstalk.ophys_plane import DecrosstalkingOphysPlane


def test_plane_instantiation(ophys_plane_data_fixture):
    schema_dict = ophys_plane_data_fixture
    ct = 0
    for pair in schema_dict['coupled_planes']:
        for plane_args in pair['planes']:
            _ = DecrosstalkingOphysPlane.from_schema_dict(plane_args)
            ct += 1
    assert ct == 4


def test_setting_qc_path(ophys_plane_data_fixture):
    schema_dict = ophys_plane_data_fixture
    plane_data = schema_dict['coupled_planes'][0]['planes'][0]
    plane = DecrosstalkingOphysPlane.from_schema_dict(plane_data)
    assert plane.qc_file_path is None
    plane.qc_file_path = 'path/to/a/file'
    assert plane.qc_file_path == 'path/to/a/file'
