import pytest
import copy
import tifffile
import pathlib
import h5py
from unittest.mock import patch
import tempfile
import numpy as np
from ophys_etl.utils.array_utils import normalize_array
from ophys_etl.modules.mesoscope_splitting.__main__ import (
    TiffSplitterCLI)


@pytest.fixture(scope='session')
def splitter_tmp_dir_fixture(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp('splitter_cli_test')
    return pathlib.Path(tmp_dir)


@pytest.fixture(scope='session')
def z_to_exp_id_fixture():
    """
    dict mapping z value to ophys_experiment_id
    """
    return {2: 222, 1: 111, 3: 333, 4: 444, 5: 555,
            6: 666, 7: 777, 8: 888}


@pytest.fixture(scope='session')
def roi_index_to_z_fixture():
    return {0: [1, 2, 3, 4], 1: [5, 6, 7, 8]}


@pytest.fixture(scope='session')
def z_to_roi_index_fixture():
    return {1: 0, 2: 0, 3: 0, 4: 0,
            5: 1, 6: 1, 7: 1, 8: 1}


@pytest.fixture(scope='session')
def z_to_stack_path_fixture(splitter_tmp_dir_fixture):
    tmp_dir = splitter_tmp_dir_fixture

    path_0 = tmp_dir / 'z_stack_0.tiff'
    path_0 = str(path_0.resolve().absolute())

    path_1 = tmp_dir / 'z_stack_1.tiff'
    path_1 = str(path_1.resolve().absolute())

    path_2 = tmp_dir / 'z_stack_2.tiff'
    path_2 = str(path_2.resolve().absolute())

    path_3 = tmp_dir / 'z_stack_3.tiff'
    path_3 = str(path_3.resolve().absolute())

    return {1: path_0, 2: path_0,
            3: path_1, 4: path_1,
            5: path_2, 6: path_2,
            7: path_3, 8: path_3}


@pytest.fixture(scope='session')
def image_metadata_fixture():
    """
    List of dicts representing metadata for depth and
    timeseries tiffs
    """

    z_list = [[2, 1], [3, 4], [6, 5], [8, 7]]

    metadata = []
    si_metadata = dict()
    si_metadata['SI.hStackManager.zsAllActuators'] = z_list
    metadata.append(si_metadata)

    roi_list = []
    roi_list.append({'zs': [1, 2, 3, 4],
                     'scanfields': [{'centerXY': [9, 10]},
                                    {'centerXY': [9, 10]}]})

    roi_list.append({'zs': [5, 6, 7, 8],
                     'scanfields': [{'centerXY': [11, 12]},
                                    {'centerXY': [11, 12]}]})

    metadata.append({'RoiGroups': {'imagingRoiGroup': {'rois': roi_list}}})

    return metadata


@pytest.fixture(scope='session')
def surface_metadata_fixture(image_metadata_fixture):
    """
    List of dicts representing metadata for surface TIFF
    """
    metadata = copy.deepcopy(image_metadata_fixture)
    z_list = [[29, 0], [39, 0]]
    metadata[0]['SI.hStackManager.zsAllActuators'] = z_list
    rois = metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
    rois[0]['zs'] = 29
    rois[1]['zs'] = 39
    return metadata


@pytest.fixture(scope='session')
def zstack_metadata_fixture(splitter_tmp_dir_fixture,
                            image_metadata_fixture,
                            z_to_stack_path_fixture):
    """
    Dict mapping z-stack path to metadata for the z-stack file
    """
    result = dict()
    this_path = z_to_stack_path_fixture[1]
    this_metadata = copy.deepcopy(image_metadata_fixture)
    rois = this_metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
    rois[0]['discretePlaneMode'] = 0
    rois[1]['discretePlaneMode'] = 1
    z0_list = np.linspace(1.0, 3.0, 13)
    z1_list = np.linspace(0.0, 2.0, 13)
    z_list = []
    for z0, z1 in zip(z0_list, z1_list):
        z_list.append([z0, z1])
    this_metadata[0]['SI.hStackManager.zsAllActuators'] = z_list
    result[this_path] = this_metadata

    this_path = z_to_stack_path_fixture[3]
    this_metadata = copy.deepcopy(image_metadata_fixture)
    rois = this_metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
    rois[0]['discretePlaneMode'] = 0
    rois[1]['discretePlaneMode'] = 1
    z0_list = np.linspace(2.0, 4.0, 13)
    z1_list = np.linspace(3.0, 5.0, 13)
    z_list = []
    for z0, z1 in zip(z0_list, z1_list):
        z_list.append([z0, z1])
    this_metadata[0]['SI.hStackManager.zsAllActuators'] = z_list
    result[this_path] = this_metadata

    this_path = z_to_stack_path_fixture[5]
    this_metadata = copy.deepcopy(image_metadata_fixture)
    rois = this_metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
    rois[0]['discretePlaneMode'] = 1
    rois[1]['discretePlaneMode'] = 0
    z0_list = np.linspace(5.0, 7.0, 13)
    z1_list = np.linspace(4.0, 6.0, 13)
    z_list = []
    for z0, z1 in zip(z0_list, z1_list):
        z_list.append([z0, z1])
    this_metadata[0]['SI.hStackManager.zsAllActuators'] = z_list
    result[this_path] = this_metadata

    this_path = z_to_stack_path_fixture[7]
    this_metadata = copy.deepcopy(image_metadata_fixture)
    rois = this_metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
    rois[0]['discretePlaneMode'] = 1
    rois[1]['discretePlaneMode'] = 0
    z0_list = np.linspace(7.0, 9.0, 13)
    z1_list = np.linspace(6.0, 8.0, 13)
    z_list = []
    for z0, z1 in zip(z0_list, z1_list):
        z_list.append([z0, z1])
    this_metadata[0]['SI.hStackManager.zsAllActuators'] = z_list
    result[this_path] = this_metadata

    return result


@pytest.fixture(scope='session')
def zstack_fixture(zstack_metadata_fixture,
                   z_to_exp_id_fixture,
                   splitter_tmp_dir_fixture):
    """
    Create zstack files.
    Return a dict mapping each individual experiment
    to its expected zstack as an HDF5 file.
    """
    rng = np.random.default_rng(7123412)
    tmp_dir = splitter_tmp_dir_fixture
    n_pages = 10
    exp_id_to_expected = dict()
    for tiff_path in zstack_metadata_fixture.keys():
        raw_data = dict()
        this_metadata = zstack_metadata_fixture[tiff_path]
        z_array = np.array(this_metadata[0]['SI.hStackManager.zsAllActuators'])
        z_array = z_array.transpose()
        z0 = np.mean(z_array[0, :]).astype(int)
        z1 = np.mean(z_array[1, :]).astype(int)
        for zz in (z0, z1):
            exp_id = z_to_exp_id_fixture[zz]
            expected_path = tmp_dir / f'{exp_id}_expected_z_stack.h5'
            data = rng.integers(0, 10*zz, (n_pages, 24, 24))
            raw_data[zz] = data
            exp_id_to_expected[f'expected_{exp_id}'] = expected_path
            with h5py.File(expected_path, 'w') as out_file:
                out_file.create_dataset('data', data=data)

        tiff_data = []
        for i_page in range(n_pages):
            for zz in (z0, z1):
                tiff_data.append(raw_data[zz][i_page, :, :])

        tifffile.imsave(tmp_dir / tiff_path, tiff_data)

    return exp_id_to_expected


@pytest.fixture(scope='session')
def surface_fixture(splitter_tmp_dir_fixture):
    """
    Returns path to surface tiff and path to expected tiffs
    for each ROI
    """
    tmp_dir = splitter_tmp_dir_fixture
    raw_tiff_path = tempfile.mkstemp(dir=tmp_dir,
                                     suffix='_surface.tiff')[1]
    expected_0_path = tmp_dir / 'expected_surface_0.tiff'
    expected_1_path = tmp_dir / 'expected_surface_1.tiff'
    n_pages = 7

    rng = np.random.default_rng()
    data_0 = rng.integers(0, 100, (n_pages, 24, 24))
    expected_0 = normalize_array(array=data_0.mean(axis=0))
    data_1 = rng.integers(100, 150, (n_pages, 24, 24))
    expected_1 = normalize_array(array=data_1.mean(axis=0))

    tiff_data = []
    for i_page in range(n_pages):
        tiff_data.append(data_0[i_page, :, :])
        tiff_data.append(data_1[i_page, :, :])
    tifffile.imsave(raw_tiff_path, tiff_data)
    tifffile.imsave(expected_0_path, expected_0)
    tifffile.imsave(expected_1_path, expected_1)

    return {'raw': raw_tiff_path,
            'expected_0': str(expected_0_path.resolve().absolute()),
            'expected_1': str(expected_1_path.resolve().absolute())}


@pytest.fixture(scope='session')
def timeseries_fixture(splitter_tmp_dir_fixture,
                       image_metadata_fixture,
                       z_to_exp_id_fixture):
    """
    Create timeseries tiff.
    Return path to raw tiff and the expected HDF5 files for
    each experiment.
    """
    rng = np.random.default_rng(6123512)
    tmp_dir = splitter_tmp_dir_fixture
    z_list = image_metadata_fixture[0]['SI.hStackManager.zsAllActuators']
    z_list = np.concatenate(z_list)
    z_to_data = dict()
    n_pages = 13
    for zz in z_list:
        data = rng.integers(0, 255, (n_pages, 24, 24))
        z_to_data[zz] = data
    tiff_data = []
    for i_page in range(n_pages):
        for zz in z_list:
            tiff_data.append(z_to_data[zz][i_page, :, :])
    raw_path = tempfile.mkstemp(dir=tmp_dir, suffix='_timeseries.tiff')[1]
    tifffile.imsave(raw_path, tiff_data)
    result = dict()
    result['raw'] = raw_path
    for zz in z_list:
        exp_id = z_to_exp_id_fixture[zz]
        expected_path = tmp_dir / f'expected_{exp_id}_timeseries.h5'
        with h5py.File(expected_path, 'w') as out_file:
            out_file.create_dataset('data', data=z_to_data[zz])
        result[f'expected_{exp_id}'] = str(expected_path.resolve().absolute())
    return result


@pytest.fixture(scope='session')
def depth_fixture(splitter_tmp_dir_fixture,
                  image_metadata_fixture,
                  z_to_exp_id_fixture):
    """
    Create the depth tiff.
    Return paths to raw tiff and expected output tiffs.
    """
    rng = np.random.default_rng(334422)
    tmp_dir = splitter_tmp_dir_fixture
    z_list = image_metadata_fixture[0]['SI.hStackManager.zsAllActuators']
    z_list = np.concatenate(z_list)
    z_to_data = dict()
    n_pages = 11
    for zz in z_list:
        data = rng.integers(zz, 2*zz, (n_pages, 24, 24))
        z_to_data[zz] = data
    tiff_data = []
    for i_page in range(n_pages):
        for zz in z_list:
            tiff_data.append(z_to_data[zz][i_page, :, :])
    raw_path = tempfile.mkstemp(dir=tmp_dir, suffix='_depth.tiff')[1]
    tifffile.imsave(raw_path, tiff_data)
    result = dict()
    result['raw'] = raw_path
    for zz in z_list:
        exp_id = z_to_exp_id_fixture[zz]
        expected_path = tmp_dir / f'expected_{exp_id}_depth.tiff'
        expected_data = np.mean(z_to_data[zz], axis=0)
        expected_data = normalize_array(array=expected_data)
        tifffile.imsave(expected_path, expected_data)
        result[f'expected_{exp_id}'] = str(expected_path.resolve().absolute())
    return result


@pytest.fixture(scope='session')
def input_json_fixture(
        depth_fixture,
        surface_fixture,
        timeseries_fixture,
        zstack_metadata_fixture,
        image_metadata_fixture,
        z_to_exp_id_fixture,
        z_to_stack_path_fixture,
        roi_index_to_z_fixture,
        splitter_tmp_dir_fixture,
        tmp_path_factory,
        z_to_roi_index_fixture,
        zstack_fixture):
    """
    Return dict of input data for Mesoscope TIFF splitting CLI
    """
    output_tmp_dir = tmp_path_factory.mktemp('splitter_cli_output')
    output_tmp_dir = pathlib.Path(output_tmp_dir)
    params = dict()
    params['depths_tif'] = depth_fixture['raw']
    params['surface_tif'] = surface_fixture['raw']
    params['timeseries_tif'] = timeseries_fixture['raw']
    params['storage_directory'] = str(output_tmp_dir.resolve().absolute())

    plane_groups = []
    for z_pair in ((1, 2), (3, 4), (5, 6), (7, 8)):
        this_group = dict()
        z_stack_path = z_to_stack_path_fixture[z_pair[0]]
        this_group['local_z_stack_tif'] = z_stack_path
        experiments = []
        for zz in z_pair:
            this_experiment = dict()
            exp_id = z_to_exp_id_fixture[zz]
            exp_dir = output_tmp_dir / f'{exp_id}_dir'
            if not exp_dir.exists():
                exp_dir.mkdir()
            exp_dir = str(exp_dir.resolve().absolute())
            this_experiment['experiment_id'] = exp_id
            this_experiment['storage_directory'] = exp_dir
            this_experiment['roi_index'] = z_to_roi_index_fixture[zz]
            this_experiment['scanfield_z'] = zz
            this_experiment['offset_x'] = 0
            this_experiment['offset_y'] = 0
            this_experiment['rotation'] = 22.1*zz
            this_experiment['resolution'] = 0.001*zz
            experiments.append(this_experiment)
        this_group['ophys_experiments'] = experiments
        plane_groups.append(this_group)
    params['plane_groups'] = plane_groups
    params['log_level'] = 'WARNING'
    return params


def test_splitter_cli(input_json_fixture,
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
