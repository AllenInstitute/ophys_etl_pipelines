from ophys_etl.types import OphysROI


def test_roi_instantiation(
        ophys_plane_data_fixture):
    schema_dict = ophys_plane_data_fixture
    ct = 0
    for meta_pair in schema_dict['coupled_planes']:
        pair = meta_pair['planes']
        for plane in pair:
            for roi_args in plane['rois']:
                _ = OphysROI.from_schema_dict(roi_args)
                ct += 1
    assert ct == 8
