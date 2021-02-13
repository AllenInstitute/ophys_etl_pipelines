import pytest
import tifffile
import numpy as np
import h5py
import os
import copy
from ophys_etl.transforms.mesoscope_2p import MesoscopeTiff
from ophys_etl.pipelines.brain_observatory.scripts import (
    run_mesoscope_splitting)


class MesoscopeTiffDummy(MesoscopeTiff):
    """
    A class to allow us to pass in fake metadata by hand
    """

    def __init__(self, source_tiff, cache=False):
        self._n_pages = None
        self._planes = None
        self._volumes = None
        self._source = source_tiff
        self._tiff = tifffile.TiffFile(self._source)
        self._tiff.pages.cache = cache


def generate_fake_timeseries(tmp_filename, frame_zs, roi_zs):
    """
    Writes a fake timeseries TIFF to disk.
    The time series TIFF will contain 40 512x512 TIFFs per ROI.
    Values in the TIFF will be z*10
    Returns frame and ROI metadata needed by MesoscopeTiffDummy.

    Parameters
    -----------
    tmp_filename -- path to file where we should writ the TIFF

    frame_zs -- z values of scans as they should appear
                in SI.hStackManager.zs field, i.e. [[z1, z2], [z3, z4],...]

    roi_zs -- a list of lists; each sub-list contains the z values
              for that ROI field's metadata

    Returns
    -------
    frame_metadata -- dict to be assigned to MesoscopeTiffDummy._frame_data

    roi_metadata -- dict to be assigned to MesosocopeTiffDummy._roi_data

    flattened_z -- a list of z values in the order that they should be
                   written to the TIFF
    """

    n_roi = 0
    roi_z_set = set()
    for roi_list in roi_zs:
        n_roi += len(roi_list)
        for zz in roi_list:
            roi_z_set.add(zz)

    flattened_z = []
    for frame_z_list in frame_zs:
        for zz in frame_z_list:
            if zz in roi_z_set:
                flattened_z.append(zz)

    assert len(flattened_z) == n_roi

    # generate fake tiff data and write it to tmp_filename
    tiff_data = np.zeros((n_roi*40, 512, 512), dtype=int)
    for ii in range(n_roi):
        tiff_data[ii:n_roi*40:n_roi, :, :] = 10*flattened_z[ii]

    tifffile.imwrite(tmp_filename, tiff_data, bigtiff=True)
    del tiff_data

    frame_metadata = {}
    frame_metadata['SI'] = {}
    frame_metadata['SI']['hStackManager'] = {}
    frame_metadata['SI']['hStackManager']['zs'] = copy.deepcopy(frame_zs)
    frame_metadata['SI']['hFastZ'] = {}
    frame_metadata['SI']['hFastZ']['userZs'] = copy.deepcopy(frame_zs)

    frame_metadata['SI']['hChannels'] = {}
    frame_metadata['SI']['hChannels']['channelsActive'] = [[1], [2]]

    _rois = []
    for roi_list in roi_zs:
        scanfields = [{'pixelResolutionXY': (512, 512)}]*len(roi_list)

        _rois.append({'zs': copy.deepcopy(roi_list),
                      'discretePlaneMode': True,
                      'scanfields': scanfields})

    roi_metadata = {}
    roi_metadata['RoiGroups'] = {}
    roi_metadata['RoiGroups']['imagingRoiGroup'] = {}
    roi_metadata['RoiGroups']['imagingRoiGroup']['rois'] = _rois

    return frame_metadata, roi_metadata, flattened_z


def generate_experiments(flattened_z, roi_zs, storage_dir):
    """
    Parameters
    -----------
    flattened_z -- a list of z values for the scan plane

    roi_zs -- a list of lists; each sub-list contains the z values
              for that ROI field's metadata

    storage_dir -- path to the parent of each experiment's storage_dir

    Returns
    -------
    A list of experiments suitable for passing into split_timeseries

    """
    # generate some fake experiments based on the data we put in
    # the TIFF file
    experiments = []
    for exp_id, zz in enumerate(flattened_z):

        roi_idx = None
        for i_roi, roi_list in enumerate(roi_zs):
            if zz in roi_list:
                roi_idx = i_roi
                break
        assert i_roi is not None

        exp = {}
        local_storage = os.path.join(storage_dir, 'exp_%d' % exp_id)
        if not os.path.exists(local_storage):
            os.makedirs(local_storage)
        exp['storage_directory'] = local_storage
        exp['experiment_id'] = exp_id
        exp['roi_index'] = roi_idx
        exp['scanfield_z'] = zz

        # I do not think that this metadata gets used to split the file;
        # it is just passed through as metadata
        exp['resolution'] = 0
        exp['offset_x'] = 0
        exp['offset_y'] = 0
        exp['rotation'] = 0
        exp['height'] = 0
        exp['width'] = 0
        experiments.append(exp)

    return experiments


def validate_timeseries_split(experiment_list, storage_dir):
    """
    Actually validate that outputs were correctly written.

    Parameters
    -----------
    experiment_list -- a list of experiments whose h5 files we
                       are validating

    storage_dir -- path to the parent of each experiment's storage_dir
    """

    # make sure the values in the HDF5 files are what we expect
    for experiment in experiment_list:
        exp_id = experiment['experiment_id']
        zz = experiment['scanfield_z']
        dirname = os.path.join(storage_dir, 'exp_%d' % exp_id)
        fname = os.path.join(dirname, '%d.h5' % exp_id)
        assert os.path.isfile(fname)
        with h5py.File(fname, 'r') as in_file:
            data = in_file['data'][()]
            assert data.shape == (40, 512, 512)
            unq = np.unique(data).flatten()
            assert unq.shape == (1,)
            assert unq[0] == zz*10


# flattened_z_expected should include all of the z values from
# frame_zs in the order that they would occur in frame_zs.flatten(),
# excluding any values that do not occur in roi_zs
@pytest.mark.parametrize("frame_zs,roi_zs,flattened_z_expected",
                         [([[22, 33], [44, 55], [66, 77], [88, 99]],
                           [[22, 44, 66, 88], [33, 55, 77, 99]],
                           [22, 33, 44, 55, 66, 77, 88, 99]),
                          ([[22, 33], [44, 55], [66, 77], [88, 99]],
                           [[22, 33, 44, 55], [66, 77, 88, 99]],
                           [22, 33, 44, 55, 66, 77, 88, 99]),
                          ([[22, 44], [66, 88], [33, 55], [77, 99]],
                           [[22, 33, 44, 55], [66, 77, 88, 99]],
                           [22, 44, 66, 88, 33, 55, 77, 99]),
                          ([[22, 0], [44, 0]],
                           [[22], [44]],
                           [22, 44]),
                          ([[44, 0], [22, 0]],
                           [[22], [44]],
                           [44, 22]),
                          ([[22, 9], [44, 6]],
                           [[22], [44]],
                           [22, 44]),
                          ([[44, 1], [22, 4]],
                           [[22], [44]],
                           [44, 22])])
def test_timeseries_split(tmpdir, frame_zs, roi_zs, flattened_z_expected):
    storage_dir = os.path.join(tmpdir, 'timeseries_storage')
    tiff_fname = os.path.join(tmpdir, 'timeseries.tiff')

    # generate mock metadata to be passed directly to
    # MesoscopeTiffDummy

    (frame_metadata,
     roi_metadata,
     flattened_z) = generate_fake_timeseries(tiff_fname,
                                             frame_zs,
                                             roi_zs)

    assert flattened_z == flattened_z_expected

    # actually read our test data from tmp

    mtiff = MesoscopeTiffDummy(tiff_fname, cache=True)
    mtiff._frame_data = frame_metadata
    mtiff._roi_data = roi_metadata

    for zz in flattened_z:
        roi_idx = None
        for ii, roi_list in enumerate(roi_zs):
            if zz in roi_list:
                roi_idx = ii
                break
        assert roi_idx is not None
        plane = mtiff.nearest_plane(roi_idx, zz)
        np.testing.assert_array_equal(np.unique(plane.asarray()),
                                      np.array([zz*10]))

    experiment_list = generate_experiments(flattened_z, roi_zs, storage_dir)

    _ = run_mesoscope_splitting.split_timeseries(mtiff, experiment_list)

    validate_timeseries_split(experiment_list, storage_dir)
