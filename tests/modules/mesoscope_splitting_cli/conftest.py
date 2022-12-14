# The fixtures in this file can be strung together to generate
# a self consistent TIFF splitting dataset provided that the following
# independent fixtures are defined:

# z_to_exp_id_fixture -- Returning a dict mapping (z0, z1) tuples to a dict
# that maps individual  values to experiment_id

# z_list_fixture -- Returning the list you would expect to find in
# SI.hStackManager.zsAllActuators in the ScanImage metadata

# roi_index_to_z_fixture -- returning a dict mapping roi_index
# to the sub-list of z_values (as represented in z_list_fixture)

# z_to_roi_index_fixture -- Returning a dict mapping a tuple of z values
# to the roi_index to which those z-values belong

# float_resolution_fixture -- an integer indicating how many decimal places
# to include in float values of z (set to 0 if zs are integers)

# For an example of how these are implemented self-consistently, see any
# of the test_*py files in this module

import pytest
import copy
import tifffile
import pathlib
import h5py
import tempfile
import shutil
import numpy as np
import json

from ophys_etl.test_utils.full_field_tiff_utils import (
    _create_full_field_tiff,
    _create_roi_metadata)

from ophys_etl.utils.array_utils import normalize_array


def _roi_index_to_centerxy(roi_index):
    """
    These centers are in physical coordinates
    """

    center_xy_list = [
        (0.1, 0.1),
        (0.3, 0.3),
        (0.5, 0.5),
        (0.1, 0.5),
        (0.3, 0.5)]

    jj = roi_index % len(center_xy_list)
    return center_xy_list[jj]


@pytest.fixture
def full_field_roi_size_fixture():
    """
    Return a tuple representing the size of the ROIs that make up
    the full field stitched image in physical coordinates
    """
    return (1.0, 1.0)


@pytest.fixture
def surface_roi_resolutionxy_fixture():
    """
    pixelResolutionXY for the average surface image ROIs
    (these are the sizes of the ROIs in pixel coordinate)
    """

    # Note: if the size gets reduced to (4, 4),
    # then tifffile will save all of the average_surface
    # data to one page, which fouls up the tiff splitting
    # module, causing the test to fail

    return [5, 5]


@pytest.fixture
def surface_roi_sizexy_fixture():
    """
    sizeXY for the average surface image ROIs
    (these are the sizes of the ROIs in physical coordinates)
    """
    return [0.1, 0.1]


@pytest.fixture
def z_to_stack_path_fixture(splitter_tmp_dir_fixture,
                            z_list_fixture):
    """
    Return a dict mapping a tuple of z-values to the path
    to local_z_stack.tiff corresponding to those z-values
    """
    tmp_dir = splitter_tmp_dir_fixture
    result = dict()
    for pair in z_list_fixture:
        path = tempfile.mkstemp(dir=tmp_dir,
                                suffix='_z_stack.tiff')[1]
        result[tuple(pair)] = path

    return result


@pytest.fixture
def image_metadata_fixture(z_list_fixture,
                           roi_index_to_z_fixture):
    """
    ScanImage metadata for depth and timeseries TIFFs
    """

    z_list = z_list_fixture

    metadata = []
    si_metadata = dict()
    si_metadata['SI.hStackManager.zsAllActuators'] = z_list
    si_metadata['SI.hChannels.channelSave'] = [1, 2]
    metadata.append(si_metadata)

    roi_list = []
    roi_index_list = list(roi_index_to_z_fixture.keys())
    roi_index_list.sort()
    for roi_index in roi_index_list:
        zs = roi_index_to_z_fixture[roi_index]
        centerxy = _roi_index_to_centerxy(roi_index)
        this_roi = {'zs': zs,
                    'scanfields': [{'centerXY': [centerxy[0], centerxy[1]]},
                                   {'centerXY': [centerxy[0], centerxy[1]]}]}
        roi_list.append(this_roi)

    if len(roi_list) == 1:
        roi_list = roi_list[0]

    metadata.append({'RoiGroups': {'imagingRoiGroup': {'rois': roi_list}}})

    return metadata


@pytest.fixture
def surface_metadata_fixture(image_metadata_fixture,
                             roi_index_to_z_fixture,
                             surface_roi_resolutionxy_fixture,
                             surface_roi_sizexy_fixture):
    """
    ScanImage metadata for surface TIFFs
    """
    metadata = copy.deepcopy(image_metadata_fixture)

    metadata[0].pop('SI.hChannels.channelSave')
    metadata[0]['SI.hChannels.channelSave'] = 1

    n_rois = len(roi_index_to_z_fixture)
    z_list = []
    rois = metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
    for ii in range(n_rois):
        val = 1+17*ii
        z_list.append([val, 0])
        if n_rois > 1:
            this_roi = rois[ii]
        else:
            this_roi = rois

        this_roi['zs'] = val

        for scanfield in this_roi['scanfields']:
            scanfield['pixelResolutionXY'] = surface_roi_resolutionxy_fixture
            scanfield['sizeXY'] = surface_roi_sizexy_fixture

    metadata[0]['SI.hStackManager.zsAllActuators'] = z_list
    return metadata


@pytest.fixture
def zstack_metadata_fixture(splitter_tmp_dir_fixture,
                            image_metadata_fixture,
                            z_to_stack_path_fixture,
                            z_to_roi_index_fixture,
                            roi_index_to_z_fixture):
    """
    Dict mapping z-stack path to ScanImage metadata for
    the local_z_stack TIFFs
    """
    n_rois = len(roi_index_to_z_fixture)
    result = dict()
    for pair in z_to_stack_path_fixture:
        this_path = z_to_stack_path_fixture[pair]
        this_metadata = copy.deepcopy(image_metadata_fixture)
        rois = this_metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
        if n_rois > 1:
            for ii in range(len(rois)):
                rois[ii]['discretePlaneMode'] = 1
            roi_index = z_to_roi_index_fixture[tuple(pair)]
            rois[roi_index]['discretePlaneMode'] = 0
        else:
            rois['discretePlaneMode'] = 0
        z0_list = np.linspace(pair[0]-1, pair[0]+1, 13)
        z1_list = np.linspace(pair[1]-1, pair[1]+1, 13)
        z_list = []
        for z0, z1 in zip(z0_list, z1_list):
            z_list.append([z0, z1])
        this_metadata[0]['SI.hStackManager.zsAllActuators'] = z_list
        result[this_path] = this_metadata

    return result


@pytest.fixture
def zstack_fixture(zstack_metadata_fixture,
                   z_to_exp_id_fixture,
                   splitter_tmp_dir_fixture,
                   float_resolution_fixture):
    """
    Create z-stack tiff files at paths specified in
    zstack_metadata_fixture

    Also write h5 files containing the expected z-stacks
    for individual experiments. Return a dict mapping experiment_id
    to the expected h5 file.
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
        z0 = np.mean(z_array[0, :])
        z1 = np.mean(z_array[1, :])
        for zz in (z0, z1):
            if float_resolution_fixture > 0:
                z_pair = (np.round(z0, decimals=float_resolution_fixture),
                          np.round(z1, decimals=float_resolution_fixture))
                zz = np.round(zz, decimals=float_resolution_fixture)
            else:
                z_pair = (z0, z1)
                zz = zz
            exp_id = z_to_exp_id_fixture[z_pair][zz]
            expected_path = tmp_dir / f'{exp_id}_expected_z_stack.h5'
            data = rng.integers(0, 10*zz, (n_pages, 25, 24))
            raw_data[zz] = data
            exp_id_to_expected[f'expected_{exp_id}'] = expected_path
            with h5py.File(expected_path, 'w') as out_file:
                out_file.create_dataset('data', data=data)

        tiff_data = []
        for i_page in range(n_pages):
            for zz in (z0, z1):
                if float_resolution_fixture > 0:
                    zz = np.round(zz, decimals=float_resolution_fixture)
                tiff_data.append(raw_data[zz][i_page, :, :])

        tifffile.imsave(tmp_dir / tiff_path, tiff_data)

    yield exp_id_to_expected

    for exp_id in exp_id_to_expected:
        this_path = pathlib.Path(exp_id_to_expected[exp_id])
        if this_path.is_file():
            this_path.unlink()


@pytest.fixture
def surface_fixture(splitter_tmp_dir_fixture,
                    roi_index_to_z_fixture,
                    surface_roi_resolutionxy_fixture):
    """
    Create the raw surface TIFF file as well as the
    expected TIFFs associated with individual experiments.

    Return a dict mapping
    'raw' -> the path to the raw surface TIFF

    'expected_{exp_id}' -> the path to the expected TIFF
    for an individual experiment
    """
    n_rois = len(roi_index_to_z_fixture)
    tmp_dir = splitter_tmp_dir_fixture
    raw_tiff_path = tempfile.mkstemp(dir=tmp_dir,
                                     suffix='_surface.tiff')[1]
    expected_path_list = []
    for ii in range(n_rois):
        expected_path = tmp_dir / f'expected_surface_{ii}.tiff'
        expected_path_list.append(expected_path)
    n_pages = 7

    data_list = []
    expected_img_list = []

    rng = np.random.default_rng(554433)
    for ii in range(n_rois):

        this_data = rng.integers(0, 100,
                                 (n_pages,
                                  surface_roi_resolutionxy_fixture[0],
                                  surface_roi_resolutionxy_fixture[1]))

        this_expected = normalize_array(this_data.mean(axis=0))
        data_list.append(this_data)
        expected_img_list.append(this_expected)

    tiff_data = []
    for i_page in range(n_pages):
        for ii in range(n_rois):
            tiff_data.append(data_list[ii][i_page, :, :])

    tifffile.imsave(raw_tiff_path, tiff_data)
    for expected_path, expected_img in zip(expected_path_list,
                                           expected_img_list):
        tifffile.imsave(expected_path, expected_img)

    result = dict()
    result['raw'] = raw_tiff_path
    for ii in range(n_rois):
        str_path = str(expected_path_list[ii].resolve().absolute())
        result[f'expected_{ii}'] = str_path
    yield result

    for key in result:
        this_path = pathlib.Path(result[key])
        if this_path.is_file():
            this_path.unlink()


@pytest.fixture
def timeseries_fixture(splitter_tmp_dir_fixture,
                       image_metadata_fixture,
                       z_to_exp_id_fixture):
    """
    Create the raw timeseries TIFF as well as HDF5 files
    containing the expected videos for individual experiments.

    Returns a dict mapping
    'raw' -> the path to the raw timeseries TIFF

    'expected_{exp_id}' -> path to the expected HDF5 file for a given
    experiment
    """
    rng = np.random.default_rng(6123512)
    tmp_dir = splitter_tmp_dir_fixture
    z_list = image_metadata_fixture[0]['SI.hStackManager.zsAllActuators']
    z_to_data = dict()
    n_pages = 13
    for z_pair in z_list:
        z_pair = tuple(z_pair)
        z_to_data[z_pair] = {}
        for zz in z_pair:
            data = rng.integers(0, 255, (n_pages, 25, 24))
            z_to_data[z_pair][zz] = data
    tiff_data = []
    for i_page in range(n_pages):
        for z_pair in z_list:
            z_pair = tuple(z_pair)
            for zz in z_pair:
                tiff_data.append(z_to_data[z_pair][zz][i_page, :, :])
    raw_path = tempfile.mkstemp(dir=tmp_dir, suffix='_timeseries.tiff')[1]
    tifffile.imsave(raw_path, tiff_data)
    result = dict()
    result['raw'] = raw_path
    for z_pair in z_list:
        z_pair = tuple(z_pair)
        for zz in z_pair:
            exp_id = z_to_exp_id_fixture[z_pair][zz]
            expected_path = tmp_dir / f'expected_{exp_id}_timeseries.h5'
            with h5py.File(expected_path, 'w') as out_file:
                out_file.create_dataset('data', data=z_to_data[z_pair][zz])
            str_path = str(expected_path.resolve().absolute())
            result[f'expected_{exp_id}'] = str_path
    yield result

    for key in result:
        this_path = pathlib.Path(result[key])
        if this_path.is_file():
            this_path.unlink()


@pytest.fixture
def depth_fixture(splitter_tmp_dir_fixture,
                  image_metadata_fixture,
                  z_to_exp_id_fixture):
    """
    Create the raw depth TIFF file as well as the
    expected TIFFs associated with individual experiments.

    Return a dict mapping
    'raw' -> the path to the raw depth TIFF

    'expected_{exp_id}' -> the path to the expected TIFF
    for an individual experiment
    """
    rng = np.random.default_rng(334422)
    tmp_dir = splitter_tmp_dir_fixture
    z_list = image_metadata_fixture[0]['SI.hStackManager.zsAllActuators']
    z_to_data = dict()
    n_pages = 11
    for z_pair in z_list:
        z_pair = tuple(z_pair)
        z_to_data[z_pair] = dict()
        for zz in z_pair:
            data = rng.integers(zz, 2*zz, (n_pages, 25, 24))
            z_to_data[z_pair][zz] = data
    tiff_data = []
    for i_page in range(n_pages):
        for z_pair in z_list:
            z_pair = tuple(z_pair)
            for zz in z_pair:
                tiff_data.append(z_to_data[z_pair][zz][i_page, :, :])
    raw_path = tempfile.mkstemp(dir=tmp_dir, suffix='_depth.tiff')[1]
    tifffile.imsave(raw_path, tiff_data)
    result = dict()
    result['raw'] = raw_path
    for z_pair in z_list:
        z_pair = tuple(z_pair)
        for zz in z_pair:
            exp_id = z_to_exp_id_fixture[z_pair][zz]
            expected_path = tmp_dir / f'expected_{exp_id}_depth.tiff'
            expected_data = np.mean(z_to_data[z_pair][zz], axis=0)
            expected_data = normalize_array(array=expected_data)
            tifffile.imsave(expected_path, expected_data)
            str_path = str(expected_path.resolve().absolute())
            result[f'expected_{exp_id}'] = str_path
    yield result

    for key in result:
        this_path = pathlib.Path(result[key])
        if this_path.is_file():
            this_path.unlink()


@pytest.fixture
def full_field_2p_tiff_fixture(
        splitter_tmp_dir_fixture,
        surface_roi_sizexy_fixture,
        full_field_roi_size_fixture):
    """
    Create a test full field TIFF image


    Return a dict with
    'raw' -> the path to the raw tiff image
    'metadata' -> the metadata dict that goes with this file
    """
    gap = 1
    nrois = 3
    roix = 24
    roiy = 24

    nrows = gap*(nrois-1)+roiy*nrois
    ncols = roix

    (tiff_path,
     _,
     metadata) = _create_full_field_tiff(
        numVolumes=4,
        numSlices=3,
        seed=1123,
        output_dir=splitter_tmp_dir_fixture,
        nrows=nrows,
        ncols=ncols)

    roi_metadata = _create_roi_metadata(
            nrois=nrois,
            roix=roix,
            roiy=roiy,
            sizex=full_field_roi_size_fixture[0],
            sizey=full_field_roi_size_fixture[1])

    metadata.append(roi_metadata)

    result = {'raw': str(tiff_path.resolve().absolute()),
              'metadata': metadata}

    yield result

    if tiff_path.exists():
        tiff_path.unlink()


@pytest.fixture
def platform_json_fixture(
        splitter_tmp_dir_fixture,
        full_field_2p_tiff_fixture):
    """
    Write out a platform.json file; return the path to it
    """
    platform_path = splitter_tmp_dir_fixture / 'platform.json'
    str_path = full_field_2p_tiff_fixture['raw']
    json_data = {'fullfield_2p_image': str_path}
    with open(platform_path, 'w') as out_file:
        out_file.write(json.dumps(json_data))

    yield platform_path

    if platform_path.is_file():
        platform_path.unlink()


@pytest.fixture
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
        zstack_fixture,
        platform_json_fixture):
    """
    Return dict of input data for Mesoscope TIFF splitting CLI
    """
    output_tmp_dir = tmp_path_factory.mktemp('splitter_cli_output')
    output_tmp_dir = pathlib.Path(output_tmp_dir)
    timeseries_tmp_dir = pathlib.Path(
            tmp_path_factory.mktemp('splitter_timeseries_temps'))
    params = dict()
    params['dump_every'] = 5  # timeseries all have 13 frames
    params['depths_tif'] = depth_fixture['raw']
    params['surface_tif'] = surface_fixture['raw']
    params['timeseries_tif'] = timeseries_fixture['raw']
    params['storage_directory'] = str(output_tmp_dir.resolve().absolute())
    params['tmp_dir'] = str(timeseries_tmp_dir.resolve().absolute())
    params['platform_json_path'] = str(
            platform_json_fixture.resolve().absolute())
    params['data_upload_dir'] = str(
            splitter_tmp_dir_fixture.resolve().absolute())

    plane_groups = []
    for z_pair in z_to_stack_path_fixture:
        this_group = dict()
        z_stack_path = z_to_stack_path_fixture[z_pair]
        this_group['local_z_stack_tif'] = z_stack_path
        experiments = []
        z_pair = tuple(z_pair)
        for zz in z_pair:
            this_experiment = dict()
            exp_id = z_to_exp_id_fixture[z_pair][zz]
            exp_dir = output_tmp_dir / f'{exp_id}_dir'
            if not exp_dir.exists():
                exp_dir.mkdir()
            exp_dir = str(exp_dir.resolve().absolute())
            this_experiment['experiment_id'] = exp_id
            this_experiment['storage_directory'] = exp_dir
            this_experiment['roi_index'] = z_to_roi_index_fixture[z_pair]
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
    yield params

    ts_path = pathlib.Path(params['timeseries_tif'])
    if ts_path.is_file():
        ts_path.unlink()
    depth_path = pathlib.Path(params['depths_tif'])
    if depth_path.is_file():
        depth_path.unlink()
    surface_path = pathlib.Path(params['surface_tif'])
    if surface_path.is_file():
        surface_path.unlink()

    for plane_group in params['plane_groups']:
        zstack_path = pathlib.Path(plane_group['local_z_stack_tif'])
        if zstack_path.is_file():
            zstack_path.unlink()
        for experiment in plane_group['ophys_experiments']:
            this_dir = pathlib.Path(experiment['storage_directory'])
            path_list = [n for n in this_dir.rglob('*')]
            for this_path in path_list:
                suffix = this_path.suffix
                if suffix == '.h5' or suffix == '.tiff' or suffix == '.tif':
                    if this_path.is_file():
                        this_path.unlink()

    if timeseries_tmp_dir.exists():
        shutil.rmtree(timeseries_tmp_dir)
