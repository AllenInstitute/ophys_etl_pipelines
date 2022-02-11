from ophys_etl.types import OphysROI
import numpy as np


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


def test_roi_global_pixel_set():
    width = 7
    height = 5
    mask = np.zeros((height, width), dtype=bool)
    mask[2, 4] = True
    mask[3, 6] = True
    roi = OphysROI(roi_id=1,
                   x0=100,
                   y0=200,
                   width=width,
                   height=height,
                   valid_roi=True,
                   mask_matrix=mask)
    assert roi.global_pixel_set == set([(202, 104), (203, 106)])
