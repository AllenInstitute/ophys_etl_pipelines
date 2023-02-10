# This file tests the mesoscope tiff splitting CLI for differen
# 'flavors' of configuration. The 'flavors' are:
#
# '4x2' -- 4 ROIs x 2 planes
# '4x2_repeat' -- 4 ROIs x 2 planes with a z value repeated in many ROIs
# '4x2_floats' -- 4 ROIs x2 planes; z-values are floats
# '2x4' -- 2 ROIS x 4 planes
# '2x4_repeats' -- 2 ROIs x 4planes with a z value repeated in many ROIs
# '1x6' -- 1 ROI x 6 planes

import pytest
import pathlib
import copy
from itertools import product
from utils import run_mesoscope_cli_test


@pytest.fixture
def flavor(request):
    """
    This fixture is how we will pass around the type
    of configuration we are testing (1x6, 2x4, 2x4_with_repeats, etc.)
    """
    return request.param


@pytest.fixture
def float_resolution_fixture(flavor):
    if flavor in ('1x6', '4x2', '2x4_repeats', '2x4', '4x2_repeats'):
        return 0
    elif flavor == '4x2_floats':
        return 1


@pytest.fixture
def splitter_tmp_dir_fixture(flavor, tmp_path_factory, helper_functions):
    tmp_dir = tmp_path_factory.mktemp(f'mesoscope_testing_{flavor}_')
    yield tmp_dir
    helper_functions.clean_up_dir(tmp_dir)


@pytest.fixture
def z_to_exp_id_fixture(flavor):
    """
    Returning a dict mapping (z0, z1) tuples to a dict
    that maps individual  values to experiment_id
    """
    if flavor == '1x6':
        return {(2, 1): {2: 222, 1: 111},
                (3, 4): {3: 333, 4: 444},
                (6, 5): {6: 666, 5: 555}}
    elif flavor == '4x2':
        return {(2, 1): {2: 222, 1: 111},
                (3, 4): {3: 333, 4: 444},
                (6, 5): {6: 666, 5: 555},
                (8, 7): {8: 888, 7: 777}}
    elif flavor == '4x2_floats':
        return {(2.2, 1.3): {2.2: 222, 1.3: 111},
                (3.1, 2.3): {3.1: 333, 2.3: 444},
                (1.3, 5.0): {1.3: 666, 5.0: 555},
                (8.2, 3.1): {8.2: 888, 3.1: 777}}
    elif flavor == '2x4_repeats':
        return {(2, 1): {2: 222, 1: 111},
                (3, 4): {3: 333, 4: 444},
                (6, 2): {6: 666, 2: 555},
                (8, 7): {8: 888, 7: 777}}
    elif flavor == '2x4':
        return {(2, 1): {2: 222, 1: 111},
                (3, 4): {3: 333, 4: 444},
                (6, 5): {6: 666, 5: 555},
                (8, 7): {8: 888, 7: 777}}
    elif flavor == '4x2_repeats':
        return {(2, 1): {2: 222, 1: 111},
                (3, 2): {3: 333, 2: 444},
                (2, 5): {2: 666, 5: 555},
                (8, 2): {8: 888, 2: 777}}


@pytest.fixture
def z_list_fixture(flavor):
    """
    Return the list you would expect to find in
    SI.hStackManager.zsAllActuators in the ScanImage metadata
    """
    if flavor == '1x6':
        z_list = [[2, 1], [3, 4], [6, 5]]
    elif flavor == '4x2':
        z_list = [[2, 1], [3, 4], [6, 5], [8, 7]]
    elif flavor == '4x2_floats':
        z_list = [[2.2, 1.3], [3.1, 2.3], [1.3, 5.0], [8.2, 3.1]]
    elif flavor == '2x4_repeats':
        z_list = [[2, 1], [3, 4], [6, 2], [8, 7]]
    elif flavor == '2x4':
        z_list = [[2, 1], [3, 4], [6, 5], [8, 7]]
    elif flavor == '4x2_repeats':
        z_list = [[2, 1], [3, 2], [2, 5], [8, 2]]

    return z_list


@pytest.fixture
def roi_index_to_z_fixture(flavor):
    """
    Return a dict mapping roi_index
    to the sub-list of z_values (as represented in z_list_fixture)
    """
    if flavor == '1x6':
        return {0: [1, 2, 3, 4, 5, 6]}
    elif flavor == '4x2_floats':
        return {0: [1.3, 2.2],
                1: [2.3, 3.1],
                2: [1.3, 5.0],
                3: [3.1, 8.2]}
    elif flavor == '2x4_repeats':
        return {0: [1, 2, 3, 4], 1: [2, 6, 7, 8]}
    elif flavor == '2x4':
        return {0: [1, 2, 3, 4], 1: [5, 6, 7, 8]}
    elif flavor == '4x2':
        return {0: [1, 2],
                1: [3, 4],
                2: [5, 6],
                3: [7, 8]}
    elif flavor == '4x2_repeats':
        return {0: [1, 2],
                1: [2, 3],
                2: [2, 5],
                3: [2, 8]}


@pytest.fixture
def z_to_roi_index_fixture(flavor):
    """
    Returning a dict mapping a tuple of z values
    to the roi_index to which those z-values belong
    """
    if flavor == '1x6':
        return {(2, 1): 0,
                (3, 4): 0,
                (6, 5): 0}
    elif flavor == '4x2_floats':
        return {(2.2, 1.3): 0,
                (3.1, 2.3): 1,
                (1.3, 5.0): 2,
                (8.2, 3.1): 3}
    elif flavor == '2x4_repeats':
        return {(2, 1): 0,
                (3, 4): 0,
                (6, 2): 1,
                (8, 7): 1}
    elif flavor == '2x4':
        return {(2, 1): 0,
                (3, 4): 0,
                (6, 5): 1,
                (8, 7): 1}
    elif flavor == '4x2':
        return {(2, 1): 0,
                (3, 4): 1,
                (6, 5): 2,
                (8, 7): 3}
    elif flavor == '4x2_repeats':
        return {(2, 1): 0,
                (3, 2): 1,
                (2, 5): 2,
                (8, 2): 3}


@pytest.fixture
def expected_count(flavor):
    """
    The number of ophys experiments that should have been
    written out
    """
    if flavor == '1x6':
        return 6
    elif flavor == '4x2':
        return 8
    elif flavor == '4x2_floats':
        return 8
    elif flavor == '2x4_repeats':
        return 8
    elif flavor == '2x4':
        return 8
    elif flavor == '4x2_repeats':
        return 8


@pytest.mark.parametrize(
        'use_platform_json, use_data_upload_dir, flavor',
        product((True, False, None),
                (True, False, None),
                ('1x6', '4x2', '4x2_floats',
                 '2x4_repeats', '2x4',
                 '4x2_repeats')),
        indirect=['flavor'])
def test_splitter_cli(input_json_fixture,
                      tmp_path_factory,
                      zstack_metadata_fixture,
                      surface_metadata_fixture,
                      image_metadata_fixture,
                      depth_fixture,
                      surface_fixture,
                      timeseries_fixture,
                      zstack_fixture,
                      full_field_2p_tiff_fixture,
                      use_platform_json,
                      use_data_upload_dir,
                      expected_count,
                      flavor,
                      helper_functions):

    input_json_data = copy.deepcopy(input_json_fixture)
    expect_full_field = True
    if use_platform_json is None:
        input_json_data['platform_json_path'] = None
        expect_full_field = False
    elif not use_platform_json:
        input_json_data.pop('platform_json_path')
        expect_full_field = False

    if use_data_upload_dir is None:
        input_json_data['data_upload_dir'] = None
        expect_full_field = False
    elif not use_data_upload_dir:
        input_json_data.pop('data_upload_dir')
        expect_full_field = False

    tmp_dir = tmp_path_factory.mktemp('cli_output_json_1x6')
    tmp_dir = pathlib.Path(tmp_dir)
    actual_count = run_mesoscope_cli_test(
                input_json=input_json_data,
                tmp_dir=tmp_dir,
                zstack_metadata=zstack_metadata_fixture,
                surface_metadata=surface_metadata_fixture,
                image_metadata=image_metadata_fixture,
                depth_data=depth_fixture,
                surface_data=surface_fixture,
                timeseries_data=timeseries_fixture,
                zstack_data=zstack_fixture,
                full_field_2p_tiff_data=full_field_2p_tiff_fixture,
                expect_full_field=expect_full_field)

    assert actual_count == expected_count
    helper_functions.clean_up_dir(tmp_dir)
