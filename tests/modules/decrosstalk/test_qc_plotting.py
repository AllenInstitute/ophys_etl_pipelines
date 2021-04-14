from ophys_etl.modules.decrosstalk.ophys_plane import OphysROI
from ophys_etl.modules.decrosstalk.qc_plotting import get_roi_pixels
from ophys_etl.modules.decrosstalk.qc_plotting import find_overlapping_roi_pairs  # noqa: E501


def test_get_roi_pixels():
    """
    Test method that maps a list of ROIs to a dict of pixels
    """

    roi_list = []
    expected_pixel_set = {}

    roi = OphysROI(x0=4,
                   y0=17,
                   width=5,
                   height=3,
                   mask_matrix=[[True, False, False, True, True],
                                [False, False, False, True, False],
                                [True, True, False, False, False]],
                   roi_id=0,
                   valid_roi=True)

    roi_list.append(roi)
    expected_pixel_set[0] = set([(4, 17), (7, 17), (8, 17), (7, 18),
                                 (4, 19), (5, 19)])

    roi = OphysROI(x0=9,
                   y0=7,
                   width=2,
                   height=5,
                   mask_matrix=[[False, False],
                                [True, False],
                                [False, True],
                                [False, False],
                                [True, True]],
                   roi_id=1,
                   valid_roi=True)

    roi_list.append(roi)
    expected_pixel_set[1] = set([(9, 8), (10, 9), (9, 11), (10, 11)])

    result = get_roi_pixels(roi_list)

    assert len(result) == 2
    assert 0 in result
    assert 1 in result

    assert result[0] == expected_pixel_set[0]
    assert result[1] == expected_pixel_set[1]


def test_find_overlapping_roi_pairs():

    roi_list_0 = []
    roi_list_1 = []

    roi = OphysROI(x0=4, y0=5,
                   width=4, height=3,
                   mask_matrix=[[True, False, False, True],
                                [False, True, True, False],
                                [False, True, False, False]],
                   roi_id=0,
                   valid_roi=True)

    roi_list_0.append(roi)

    # photo-negative of roi_0
    roi = OphysROI(x0=4, y0=5,
                   width=4, height=3,
                   mask_matrix=[[False, True, True, False],
                                [True, False, False, True],
                                [True, False, True, True]],
                   roi_id=1,
                   valid_roi=True)

    roi_list_1.append(roi)

    # intersects one point of roi_0
    roi = OphysROI(x0=6, y0=6,
                   width=2, height=3,
                   mask_matrix=[[True, False],
                                [True, True],
                                [True, True]],
                   roi_id=2,
                   valid_roi=True)

    roi_list_1.append(roi)

    # no intersection with roi_0
    roi = OphysROI(x0=6, y0=6,
                   width=2, height=3,
                   mask_matrix=[[False, False],
                                [True, True],
                                [True, True]],
                   roi_id=3,
                   valid_roi=True)

    roi_list_1.append(roi)

    # one corner overlaps with roi_2 and roi_3
    roi = OphysROI(x0=7, y0=8,
                   width=3, height=4,
                   mask_matrix=[[True, True, False],
                                [True, True, True],
                                [True, True, True],
                                [True, True, True]],
                   roi_id=4,
                   valid_roi=True)

    roi_list_0.append(roi)

    # no overlaps
    roi = OphysROI(x0=7, y0=8,
                   width=3, height=4,
                   mask_matrix=[[False, False, False],
                                [True, True, True],
                                [True, True, True],
                                [True, True, True]],
                   roi_id=5,
                   valid_roi=True)

    roi_list_0.append(roi)

    overlap_list = find_overlapping_roi_pairs(roi_list_0,
                                              roi_list_1)

    assert len(overlap_list) == 3
    assert (0, 2, 1/5, 1/5) in overlap_list
    assert (4, 3, 1/11, 1/4) in overlap_list
    assert (4, 2, 1/11, 1/5) in overlap_list
