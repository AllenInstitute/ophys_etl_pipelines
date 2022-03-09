import pytest
import copy
import tifffile
import pathlib
import h5py
from unittest.mock import patch
import numpy as np
from ophys_etl.modules.mesoscope_splitting.__main__ import (
    TiffSplitterCLI)


@pytest.fixture
def splitter_tmp_dir_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('splitter_cli_test_repeats')
    return pathlib.Path(tmp_dir)


@pytest.fixture
def z_to_exp_id_fixture():
    """
    dict mapping z value to ophys_experiment_id
    """
    return {(2, 1): {2: 222, 1: 111},
            (3, 4): {3: 333, 4: 444},
            (6, 2): {6: 666, 2: 555},
            (8, 7): {8: 888, 7: 777}}


@pytest.fixture
def z_list_fixture():
    z_list = [[2, 1], [3, 4], [6, 2], [8, 7]]
    return z_list


@pytest.fixture
def roi_index_to_z_fixture():
    return {0: [1, 2, 3, 4], 1: [2, 6, 7, 8]}


@pytest.fixture
def z_to_roi_index_fixture():
    return {(2, 1): 0,
            (3, 4): 0,
            (6, 2): 1,
            (8, 7): 1}


def test_splitter_2x4_repeats(
                      input_json_fixture,
                      tmp_path_factory,
                      zstack_metadata_fixture,
                      surface_metadata_fixture,
                      image_metadata_fixture,
                      depth_fixture,
                      surface_fixture,
                      timeseries_fixture,
                      zstack_fixture):

    tmp_dir = tmp_path_factory.mktemp('cli_output_json')
    output_json = pathlib.Path(tmp_dir) / 'output.json'
    output_json = str(output_json.resolve().absolute())
    input_json = copy.deepcopy(input_json_fixture)
    input_json['output_json'] = output_json

    metadata_lookup = copy.deepcopy(zstack_metadata_fixture)
    metadata_lookup[input_json['timeseries_tif']] = image_metadata_fixture
    metadata_lookup[input_json['depths_tif']] = image_metadata_fixture
    metadata_lookup[input_json['surface_tif']] = surface_metadata_fixture

    def mock_read_metadata(tiff_path):
        str_path = str(tiff_path.resolve().absolute())
        return metadata_lookup[str_path]

    to_replace = 'ophys_etl.modules.mesoscope_splitting.'
    to_replace += 'tiff_metadata._read_metadata'
    with patch(to_replace,
               new=mock_read_metadata):

        runner = TiffSplitterCLI(
                    args=[],
                    input_data=input_json)
        runner.run()

    exp_ct = 0
    for plane_group in input_json['plane_groups']:
        for exp in plane_group['ophys_experiments']:
            exp_ct += 1
            exp_dir = exp['storage_directory']
            exp_dir = pathlib.Path(exp_dir)
            exp_id = exp['experiment_id']
            ts_actual = exp_dir / f'{exp_id}.h5'
            with h5py.File(ts_actual, 'r') as in_file:
                actual = in_file['data'][()]
            ts_expected = timeseries_fixture[f'expected_{exp_id}']
            with h5py.File(ts_expected, 'r') as in_file:
                expected = in_file['data'][()]
            np.testing.assert_array_equal(actual, expected)

            depth_actual = exp_dir / f'{exp_id}_depth.tif'
            with tifffile.TiffFile(depth_actual, 'rb') as in_file:
                assert len(in_file.pages) == 1
                actual = in_file.pages[0].asarray()
            depth_expected = depth_fixture[f'expected_{exp_id}']
            with tifffile.TiffFile(depth_expected, 'rb') as in_file:
                assert len(in_file.pages) == 1
                expected = in_file.pages[0].asarray()
            np.testing.assert_array_equal(actual, expected)

            surface_actual = exp_dir / f'{exp_id}_surface.tif'
            with tifffile.TiffFile(surface_actual, 'rb') as in_file:
                assert len(in_file.pages) == 1
                actual = in_file.pages[0].asarray()

            roi_index = exp['roi_index']
            surface_expected = surface_fixture[f'expected_{roi_index}']
            with tifffile.TiffFile(surface_expected, 'rb') as in_file:
                assert len(in_file.pages) == 1
                expected = in_file.pages[0].asarray()
            np.testing.assert_array_equal(actual, expected)

            stack_actual = exp_dir / f'{exp_id}_z_stack_local.h5'
            with h5py.File(stack_actual, 'r') as in_file:
                actual = in_file['data'][()]
            stack_expected = zstack_fixture[f'expected_{exp_id}']
            with h5py.File(stack_expected, 'r') as in_file:
                expected = in_file['data'][()]
            np.testing.assert_array_equal(actual, expected)

    assert exp_ct == 8
