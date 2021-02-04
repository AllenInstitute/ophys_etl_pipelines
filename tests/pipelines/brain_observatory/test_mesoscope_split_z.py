import pytest
import tifffile
import numpy as np
import h5py
import os
import copy
from ophys_etl.transforms.mesoscope_2p import MesoscopeTiff
from ophys_etl.pipelines.brain_observatory.scripts.run_mesoscope_splitting import split_z  # noqa: E501


"""
This file is meant to test our code which splits the *_local_z_stack*.tiff
files. This comment block contains what I have learned about the data model
underlying those files, mostly by poking around in examples and seeing how
they move through the code.

These files represent repeated scans of planes centered on planes of
interest in the brain. There should be two planes of interest per file.
The actual scans in these files are at different depths, presumably
so that we can average data in three dimensions about the planes of
interest.

The TIFF file itself contains a metadata field

SI.hStackManager.zs

which is a list of lists. Each sub-list contains two elements.
The first element represents a z depth associated with the first plane.
The second element represents a z depth associated with the second plane.

For instance

SI.hStackManager.zs = [[9.9, 15.9], [10.0, 16.0], [10.1, 16.1]]

would represent a local_stack focused on planes at 10 and 16 microns.

Note that the RoiGroups.imagingRoiGroup.rois.zs metadata does not
seem to matter in the cases of these files.

Each of these files corresponds to an ROI, where, in this case, ROI
refers to an anatomical region of the brain (e.g. VISp) that was
scanned. The metadata for the ROIs is encoded in

metadata[1]['RoiGroups']['imagingRoiGroup']['rois']

where metadata is the result of running

tifffile.read_scanimage_metadata(open(local_zstack_filename, 'rb'))

The list returned by RoiGroups.imagingRoiGroup.rois contains all
of the ROIs for the mesoscope session. The ROI that this local_zstack
file actually corresponds to is flagged with the

RoiGroups.imagingRoiGroup.rois.discretePlaneMode

field. Curiously, the ROI corresponding to this zstack file
is the one with discretePlaneMode = False (presumably because
discretePlaneMode=True means "try to match this plane based on z
depth", which is not what we want to do in the volume scanning, since
each ROI will have multiple scanned z values).

The TIFF splitting code actually associates experiments with anatomical
ROIs using the experiment['roi_index'] field, which is an integer
corresponding to the ROI in

RoiGroups.imagingRoiGroup.rois

that the experiment is actually focused on. Put another way, in order
for a local_zstack file to split properly

experiment['local_z_stack']

must point to a file in which

RoiGroups.imagingRoiGroup.roi[experiment['roi_index']].discretePlaneMode

is False
"""


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


def generate_fake_z(tmp_filename, frame_zs, n_repeats, active_roi):
    """
    Writes a fake local_zstack TIFF to disk.
    Returns frame and ROI metadata needed by MesoscopeTiffDummy.

    Parameters
    -----------
    tmp_filename -- path to file where we should writ the TIFF

    frame_zs -- z values of scans as they should appear
                in SI.hStackManager.zs field, i.e. [[z1, z2], [z3, z4],...]

    n_repeats -- int; how many times to re-scan each z

    active_roi -- int; the index of the ROI with discretePlaneMode=False

    Returns
    -------
    frame_metadata -- dict to be assigned to MesoscopeTiffDummy._frame_data

    roi_metadata -- dict to be assigned to MesosocopeTiffDummy._roi_data


    Note:
    -----
    The contents for each scan in the TIFF file will be a flat field of
    10*flattened_z[ii]*n_repeats+jj
    where jj loops over n_repeats
    """

    flattened_z = []
    for frame_z_list in frame_zs:
        for zz in frame_z_list:
            flattened_z.append(zz)

    n_z = len(flattened_z)

    # generate fake tiff data and write it to tmp_filename
    tiff_data = np.zeros((n_z*n_repeats, 512, 512), dtype=int)
    for jj in range(n_repeats):
        for ii in range(n_z):
            tiff_data[jj*n_z+ii, :, :] = int(10*flattened_z[ii]*n_repeats + jj)

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

    # for split z, all that actually matters is whether or not
    # discretePlaneMode is False
    # that is the plane that will be caught with volume_scanned
    _rois = []

    for ii in range(4):
        _rois.append({'zs': -1,
                      'discretePlaneMode': (ii != active_roi),
                      'scanfields': [{'pixelResolutionXY': (512, 512)}]})

    roi_metadata = {}
    roi_metadata['RoiGroups'] = {}
    roi_metadata['RoiGroups']['imagingRoiGroup'] = {}
    roi_metadata['RoiGroups']['imagingRoiGroup']['rois'] = _rois

    return frame_metadata, roi_metadata


def generate_experiments(flattened_z,
                         roi_indices,
                         storage_dir):
    """
    Parameters
    -----------
    flattened_z -- a list of z values for the experimental scan planes

    roi_indices -- a list of roi_index values for each experiment

    storage_dir -- path to the parent of each experiment's storage_dir

    Returns
    -------
    A list of experiments suitable for passing into split_timeseries

    """
    # generate some fake experiments based on the data we put in
    # the TIFF file
    experiments = []
    for exp_id, (zz, roi_idx) in enumerate(zip(flattened_z, roi_indices)):

        exp = {}
        local_storage = os.path.join(storage_dir, 'exp_%d' % exp_id)
        if not os.path.exists(local_storage):
            os.makedirs(local_storage)
        exp['storage_directory'] = local_storage
        exp['experiment_id'] = exp_id
        exp['scanfield_z'] = zz
        exp['roi_index'] = roi_idx

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


def validate_split_z(experiment_list, storage_dir, frame_zs, n_repeats):
    """
    Actually validate that outputs were correctly written.

    Parameters
    -----------
    experiment_list -- a list of experiments whose h5 files we
                       are validating

    storage_dir -- path to the parent of each experiment's storage_dir

    frame_zs -- the list of lists containing the SI.hStackManager.zs
                metadata from the big tiff file
    """

    # make sure the values in the HDF5 files are what we expect
    for experiment in experiment_list:
        exp_id = experiment['experiment_id']
        zz = experiment['scanfield_z']

        valid_zs = []
        for z_pair in frame_zs:
            if abs(z_pair[0]-zz) < abs(z_pair[1]-zz):
                valid_zs.append(z_pair[0])
            else:
                valid_zs.append(z_pair[1])

        n_z = len(valid_zs)
        assert n_z == len(frame_zs)

        dirname = os.path.join(storage_dir, 'exp_%d' % exp_id)
        fname = os.path.join(dirname, '%d_z_stack_local.h5' % exp_id)
        assert os.path.isfile(fname)
        with h5py.File(fname, 'r') as in_file:
            data = in_file['data'][()]
            assert data.shape == (n_z*n_repeats, 512, 512)
            for jj in range(n_repeats):
                for ii in range(n_z):
                    vz = valid_zs[ii]
                    val = int(10*vz)*n_repeats + jj
                    frame = data[jj*n_z+ii, :, :]
                    assert (frame == val).all()


@pytest.mark.parametrize("frame_zs,flattened_z,roi_indices,active_roi,"
                         "n_repeats,should_pass",
                         [([[1.1, 5.1], [1.2, 5.2], [1.3, 5.3]],
                           [1.2, 5.2],
                           [0, 0],
                           0,
                           15,
                           True),
                          ([[3.1, 6.1], [3.2, 6.2], [3.3, 6.3]],
                           [3.2, 6.2],
                           [3, 3],
                           3,
                           35,
                           True),
                          ([[3.1, 6.1], [3.2, 6.2], [3.3, 6.3]],
                           [3.2, 6.2],
                           [3, 3],
                           2,
                           15,
                           False)
                          ])
def test_split_z(tmpdir,
                 frame_zs,
                 flattened_z,
                 roi_indices,
                 active_roi,
                 n_repeats,
                 should_pass):
    """
    Parameters
    ----------
    frame_zs -- the z values of the volume scans in the local_zstack file

    flattened_zs -- the list of z values for the actual experimental planes

    roi_indices -- the ROI indices to be assigned to the experiments (a list)

    active_roi -- the ROI that has discretePlaneMode = False
                  in the TIFF metadata

    n_repeats -- number of time to rescan each z in frame_zs

    should_pass -- a boolean; if True, run this test as though we expect TIFF
                   splitting to succeed; if False, run this test as though
                   we expect TIFF splitting to fail, emitting a ValueError
                   (in the case where the experiments' roi_index does not
                   align with what is in the local_zstack)
    """

    storage_dir = os.path.join(tmpdir, 'zstack_storage')
    tiff_fname = os.path.join(tmpdir, 'zstack.tiff')

    # generate mock metadata to be passed directly to
    # MesoscopeTiffDummy

    (frame_metadata,
     roi_metadata) = generate_fake_z(tiff_fname,
                                     frame_zs,
                                     n_repeats,
                                     active_roi)

    # actually read our test data from tmp

    mtiff = MesoscopeTiffDummy(tiff_fname, cache=True)
    mtiff._frame_data = frame_metadata
    mtiff._roi_data = roi_metadata

    experiment_list = generate_experiments(flattened_z,
                                           roi_indices,
                                           storage_dir)

    if should_pass:
        for experiment in experiment_list:
            _ = split_z(mtiff, experiment, testing=True)

        validate_split_z(experiment_list, storage_dir, frame_zs, n_repeats)
    else:
        for experiment in experiment_list:
            with pytest.raises(ValueError):
                _ = split_z(mtiff, experiment, testing=True)
