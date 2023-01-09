import copy
import tifffile
import pathlib
import h5py
import json
from unittest.mock import patch
import numpy as np
from ophys_etl.modules.mesoscope_splitting.__main__ import (
    TiffSplitterCLI)


def run_mesoscope_cli_test(
        input_json,
        tmp_dir,
        zstack_metadata,
        surface_metadata,
        image_metadata,
        depth_data,
        surface_data,
        timeseries_data,
        zstack_data) -> int:
    """
    This is a utility method to run the mesoscope CLI test using
    data generated in a self-consistent way by the pytest fixtures
    defined in this test module.

    This method calls the mesoscope_splitting CLI and verifies that
    all expected output files were created with the correct contents
    (as defined by the fixtures passed in here).

    Parameters
    ----------
    input_json:
        The result of input_json_fixture

    tmp_dir: pathlib.Path
        temporary directory where the output.json will get written

    zstack_metadata:
        The result of zstack_metadata_fixture

    surface_metadata:
        The result of surface_metadata_fixture

    image_metadata:
        The result of surface_metadta_fixture

    depth_data:
        The result of depth_fixture

    surface_data:
        The result of surface_fixture

    timeseries_data:
        The result of timeseries_fixture

    zstack_data:
        The result of zstack_fixture

    Returns
    -------
    an integer indicating the number of simulated experiments that
    were generated and checked by this test.
    """

    output_json = tmp_dir / 'output.json'
    output_json = str(output_json.resolve().absolute())
    input_json = copy.deepcopy(input_json)
    input_json['output_json'] = output_json

    metadata_lookup = copy.deepcopy(zstack_metadata)
    metadata_lookup[input_json['timeseries_tif']] = image_metadata
    metadata_lookup[input_json['depths_tif']] = image_metadata
    metadata_lookup[input_json['surface_tif']] = surface_metadata

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
        this_zstack_path = plane_group['local_z_stack_tif']
        for exp in plane_group['ophys_experiments']:
            exp_ct += 1
            exp_dir = exp['storage_directory']
            exp_dir = pathlib.Path(exp_dir)
            exp_id = exp['experiment_id']
            ts_actual = exp_dir / f'{exp_id}.h5'
            with h5py.File(ts_actual, 'r') as in_file:
                actual_timeseries_metadata = json.loads(
                        in_file['scanimage_metadata'][()].decode('utf-8'))

                assert actual_timeseries_metadata == image_metadata

                actual = in_file['data'][()]

            ts_expected = timeseries_data[f'expected_{exp_id}']
            with h5py.File(ts_expected, 'r') as in_file:
                expected = in_file['data'][()]
            np.testing.assert_array_equal(actual, expected)

            depth_actual = exp_dir / f'{exp_id}_depth.tif'
            with tifffile.TiffFile(depth_actual, mode='rb') as in_file:
                assert len(in_file.pages) == 1
                actual = in_file.pages[0].asarray()
                actual_depth_metadata = in_file.shaped_metadata[0][
                                          'scanimage_metadata']
                assert actual_depth_metadata == image_metadata

            depth_expected = depth_data[f'expected_{exp_id}']
            with tifffile.TiffFile(depth_expected, mode='rb') as in_file:
                assert len(in_file.pages) == 1
                expected = in_file.pages[0].asarray()
            np.testing.assert_array_equal(actual, expected)

            surface_actual = exp_dir / f'{exp_id}_surface.tif'
            with tifffile.TiffFile(surface_actual, mode='rb') as in_file:
                assert len(in_file.pages) == 1
                actual = in_file.pages[0].asarray()
                actual_surface_metadata = in_file.shaped_metadata[0][
                                           'scanimage_metadata']
                assert actual_surface_metadata == surface_metadata

            roi_index = exp['roi_index']
            surface_expected = surface_data[f'expected_{roi_index}']
            with tifffile.TiffFile(surface_expected, mode='rb') as in_file:
                assert len(in_file.pages) == 1
                expected = in_file.pages[0].asarray()
            np.testing.assert_array_equal(actual, expected)

            stack_expected = zstack_data[f'expected_{exp_id}']
            stack_actual = exp_dir / f'{exp_id}_z_stack_local.h5'
            expected_zstack_metadata = zstack_metadata[this_zstack_path]
            with h5py.File(stack_actual, 'r') as in_file:
                actual = in_file['data'][()]
                actual_zstack_metadata = json.loads(
                        in_file['scanimage_metadata'][()].decode('utf-8'))
                assert actual_zstack_metadata == expected_zstack_metadata
            with h5py.File(stack_expected, 'r') as in_file:
                expected = in_file['data'][()]
            np.testing.assert_array_equal(actual, expected)
    return exp_ct
