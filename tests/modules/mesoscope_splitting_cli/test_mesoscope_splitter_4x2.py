# This file defines all of the independent data fixture necessary
# to test TIFF splitting in the case of a 4 ROI x 2 plane OPhys session

import pytest
import pathlib
from utils import run_mesoscope_cli_test


@pytest.fixture
def float_resolution_fixture():
    return 0


@pytest.fixture
def splitter_tmp_dir_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('splitter_cli_test_4x2')
    return pathlib.Path(tmp_dir)


@pytest.fixture
def z_to_exp_id_fixture():
    """
    dict mapping z value to ophys_experiment_id
    """
    return {(2, 1): {2: 222, 1: 111},
            (3, 4): {3: 333, 4: 444},
            (6, 5): {6: 666, 5: 555},
            (8, 7): {8: 888, 7: 777}}


@pytest.fixture
def z_list_fixture():
    z_list = [[2, 1], [3, 4], [6, 5], [8, 7]]
    return z_list


@pytest.fixture
def roi_index_to_z_fixture():
    return {0: [1, 2],
            1: [3, 4],
            2: [5, 6],
            3: [7, 8]}


@pytest.fixture
def z_to_roi_index_fixture():
    return {(2, 1): 0,
            (3, 4): 1,
            (6, 5): 2,
            (8, 7): 3}


def test_splitter_4x2(input_json_fixture,
                      tmp_path_factory,
                      zstack_metadata_fixture,
                      surface_metadata_fixture,
                      image_metadata_fixture,
                      depth_fixture,
                      surface_fixture,
                      timeseries_fixture,
                      zstack_fixture,
                      full_field_2p_tiff_fixture):

    tmp_dir = tmp_path_factory.mktemp('cli_output_json_4x2')
    tmp_dir = pathlib.Path(tmp_dir)
    exp_ct = run_mesoscope_cli_test(
                input_json=input_json_fixture,
                tmp_dir=tmp_dir,
                zstack_metadata=zstack_metadata_fixture,
                surface_metadata=surface_metadata_fixture,
                image_metadata=image_metadata_fixture,
                depth_data=depth_fixture,
                surface_data=surface_fixture,
                timeseries_data=timeseries_fixture,
                zstack_data=zstack_fixture,
                full_field_2p_tiff_data=full_field_2p_tiff_fixture)

    assert exp_ct == 8
