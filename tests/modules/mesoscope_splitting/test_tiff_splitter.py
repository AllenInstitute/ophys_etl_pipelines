import pytest
from typing import List
from unittest.mock import patch
import numpy as np
import h5py
import pathlib
import tempfile
import tifffile
import copy

from ophys_etl.utils.array_utils import normalize_array

from ophys_etl.modules.mesoscope_splitting.tiff_splitter import (
    ScanImageTiffSplitter,
    TimeSeriesSplitter)

from ophys_etl.modules.mesoscope_splitting.zstack_splitter import (
    ZStackSplitter)


def _create_image_tiff(tmp_dir: pathlib.Path,
                       z_value_list: List[int],
                       n_rois: int,
                       use_zs: bool = False,
                       is_surface: bool = False):
    """
    A fixture simulating a depth TIFF sampling 2 ROIs,
    4 zs at each ROI.

    if use_zs == True, then use SI.hStackManager.zs
    in place of SI.hStackManager.zsAllActuators

    if is_surface == True, SI.hStackManager.zsAllActuators
    will look like [[z0, 0], [z1, 0], [z2, 0]...]

    Returns
    -------
    the path to the TIFF

    a dict mapping roi_id, z -> normalized average image

    a dict mapping roi_id, z -> expected tiff pages

    a dict that our mock of tifffile.read_scanimage_metadata needs
    to return
    """

    avg_img_lookup = dict()
    page_lookup = dict()
    tiff_pages = []
    n_pages = 5
    n_z_per_roi = len(z_value_list)//n_rois
    for i_page in range(n_pages):
        for i_z, z_value in enumerate(z_value_list):
            i_roi = i_z//n_z_per_roi
            if (i_roi, z_value) not in page_lookup:
                page_lookup[(i_roi, z_value)] = []
            value = i_roi+z_value+i_page*len(z_value_list)
            page = np.arange(value, value+24*24).astype(np.uint16)
            page = page.reshape((24, 24))
            page[i_roi:i_roi+4, i_roi:i_roi+4] = 3
            page[i_z:i_z+2, i_z:i_z+2] = 0
            tiff_pages.append(page)
            page_lookup[(i_roi, z_value)].append(page)
    tmp_path = pathlib.Path(tempfile.mkstemp(dir=tmp_dir, suffix='tiff')[1])
    tifffile.imsave(tmp_path, tiff_pages)

    tiff_pages = np.array(tiff_pages)

    for i_z, z_value in enumerate(z_value_list):
        sub_arr = tiff_pages[i_z::len(z_value_list), :, :]
        mean_img = np.mean(sub_arr, axis=0)
        roi_id = i_z//n_z_per_roi
        avg_img_lookup[(roi_id, z_value)] = normalize_array(mean_img)

    z_array = []
    if is_surface:
        for z_value in z_value_list:
            z_array.append([z_value, 0])
    else:
        for ii in range(0, len(z_value_list), 2):
            z0 = z_value_list[ii]
            z1 = z_value_list[1+ii]
            z_array.append([z0, z1])

    metadata = []
    if use_zs:
        key_name = 'SI.hStackManager.zs'
    else:
        key_name = 'SI.hStackManager.zsAllActuators'
    metadata.append({key_name: z_array})

    roi_list = []
    for ii in range(0, len(z_value_list), n_z_per_roi):
        this_list = copy.deepcopy(list(z_value_list[ii:ii+n_z_per_roi]))
        this_list.sort()
        roi_list.append({'zs': this_list})

    roi_metadata = {'RoiGroups':
                    {'imagingRoiGroup':
                     {'rois': roi_list}}}

    metadata.append(roi_metadata)

    return (tmp_path, avg_img_lookup, page_lookup, metadata)


@pytest.mark.parametrize(
    "z_value_list, n_rois, use_zs",
    [(list(range(8)), 2, True),
     (list(range(8)), 4, True),
     (list(range(6)), 3, True),
     ((0, 2, 3, 5, 7, 4, 6, 11), 4, True),
     ((0, 2, 3, 5, 7, 4, 6, 11), 2, True),
     ((0, 2, 3, 5, 7, 4, 6, 5), 4, True),
     ((0, 2, 3, 5, 7, 4, 6, 5), 2, True),
     ((0, 4, 5, 2, 8, 7), 1, True),
     (list(range(8)), 2, False),
     (list(range(8)), 4, False),
     (list(range(6)), 3, False),
     ((0, 2, 3, 5, 7, 4, 6, 11), 4, False),
     ((0, 2, 3, 5, 7, 4, 6, 11), 2, False),
     ((0, 2, 3, 5, 7, 4, 6, 5), 4, False),
     ((0, 2, 3, 5, 7, 4, 6, 5), 2, False),
     ((0, 4, 5, 2, 8, 7), 1, False)])
def test_depth_splitter(tmp_path_factory,
                        z_value_list,
                        n_rois,
                        use_zs):
    """
    Test that, when splitting a depth TIFF, _get_pages and
    write_image_tiff behave as expected
    """

    tmp_dir = tmp_path_factory.mktemp('test_depth_2x4')

    depth_tiff = _create_image_tiff(
                    pathlib.Path(tmp_dir),
                    z_value_list,
                    n_rois,
                    use_zs=use_zs,
                    is_surface=False)

    tiff_path = depth_tiff[0]
    avg_img_lookup = depth_tiff[1]
    page_lookup = depth_tiff[2]
    metadata = depth_tiff[3]

    n_z_per_roi = len(z_value_list) // n_rois

    def scanimage_metadata_mock(file_handle):
        return metadata

    with patch('tifffile.read_scanimage_metadata',
               new=scanimage_metadata_mock):

        splitter = ScanImageTiffSplitter(tiff_path=tiff_path)
        for i_z, z_value in enumerate(z_value_list):
            i_roi = i_z//n_z_per_roi
            arr = splitter._get_pages(i_roi=i_roi,
                                      z_value=z_value)
            assert len(arr) == 5
            for i_page in range(5):
                expected = page_lookup[(i_roi, z_value)][i_page]
                np.testing.assert_array_equal(expected,
                                              arr[i_page])

            tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix='.tiff')[1]
            tmp_path = pathlib.Path(tmp_path)
            splitter.write_image_tiff(i_roi=i_roi,
                                      z_value=z_value,
                                      tiff_path=tmp_path)
            with tifffile.TiffFile(tmp_path, 'rb') as tiff_file:
                assert len(tiff_file.pages) == 1
                actual = tiff_file.pages[0].asarray()

            np.testing.assert_array_equal(actual,
                                          avg_img_lookup[(i_roi, z_value)])

            if tmp_path.is_file():
                tmp_path.unlink()
        if tiff_path.is_file():
            tiff_path.unlink()


@pytest.mark.parametrize(
    "z_value_list, n_rois, use_zs",
    [(list(range(8)), 2, True),
     (list(range(8)), 4, True),
     (list(range(6)), 3, True),
     ((0, 2, 3, 5, 7, 4, 6, 11), 4, True),
     ((0, 2, 3, 5, 7, 4, 6, 11), 2, True),
     ((0, 2, 3, 5, 7, 4, 6, 5), 4, True),
     ((0, 2, 3, 5, 7, 4, 6, 5), 2, True),
     ((0, 4, 5, 2, 8, 7), 1, True),
     (list(range(8)), 2, False),
     (list(range(8)), 4, False),
     (list(range(6)), 3, False),
     ((0, 2, 3, 5, 7, 4, 6, 11), 4, False),
     ((0, 2, 3, 5, 7, 4, 6, 11), 2, False),
     ((0, 2, 3, 5, 7, 4, 6, 5), 4, False),
     ((0, 2, 3, 5, 7, 4, 6, 5), 2, False),
     ((0, 4, 5, 2, 8, 7), 1, False)])
def test_splitter_manifest(tmp_path_factory,
                           z_value_list,
                           n_rois,
                           use_zs):
    """
    Test that the various methods characterizing legal
    combinations of i_roi and z behave as expected
    """

    tmp_dir = tmp_path_factory.mktemp('test_depth_2x4')

    depth_tiff = _create_image_tiff(
                    pathlib.Path(tmp_dir),
                    z_value_list,
                    n_rois,
                    use_zs=use_zs,
                    is_surface=False)

    tiff_path = depth_tiff[0]
    metadata = depth_tiff[3]

    n_z_per_roi = len(z_value_list) // n_rois

    def scanimage_metadata_mock(file_handle):
        return metadata

    with patch('tifffile.read_scanimage_metadata',
               new=scanimage_metadata_mock):

        splitter = ScanImageTiffSplitter(tiff_path=tiff_path)
        assert splitter.n_pages == 5*len(z_value_list)
        assert splitter.n_valid_zs == len(z_value_list)
        assert splitter.n_rois == n_rois
        valid_z_per_roi = splitter.valid_z_per_roi
        assert len(valid_z_per_roi) == n_rois
        offset = 0
        for roi in valid_z_per_roi:
            expected = set(z_value_list[offset:offset+n_z_per_roi])
            assert expected == roi
            offset += n_z_per_roi

        # check for all (i_roi, z) pairs
        assert len(splitter.roi_z_manifest) == len(z_value_list)
        for pair in splitter.roi_z_manifest:
            i_roi = pair[0]
            roi_grp = metadata[1]['RoiGroups']['imagingRoiGroup']['rois']
            if isinstance(roi_grp, dict):
                assert i_roi == 0
                roi_zs = roi_grp['zs']
            else:
                roi_zs = roi_grp[i_roi]['zs']

            if isinstance(roi_zs, int):
                assert pair[1] == roi_zs
            else:
                assert pair[1] in roi_zs

    if tiff_path.is_file():
        tiff_path.unlink()


@pytest.mark.parametrize(
    "z_value_list, use_zs",
    [(list(range(3)), True),
     (list(range(4)), True),
     (list(range(2)), True),
     ((0, 5, 4), True),
     (list(range(3)), False),
     (list(range(4)), False),
     (list(range(2)), False),
     ((0, 5, 4), False)])
def test_surface_splitter(tmp_path_factory,
                          z_value_list,
                          use_zs):
    """
    Test that, when splitting a surface TIFF, _get_pages and
    write_image_tiff behave as expected
    """

    n_rois = len(z_value_list)
    n_z_per_roi = 1

    tmp_dir = tmp_path_factory.mktemp('test_depth_2x4')

    surface_tiff = _create_image_tiff(
                    pathlib.Path(tmp_dir),
                    z_value_list,
                    n_rois,
                    use_zs=use_zs,
                    is_surface=True)

    tiff_path = surface_tiff[0]
    avg_img_lookup = surface_tiff[1]
    page_lookup = surface_tiff[2]
    metadata = surface_tiff[3]

    def scanimage_metadata_mock(file_handle):
        return metadata

    with patch('tifffile.read_scanimage_metadata',
               new=scanimage_metadata_mock):

        splitter = ScanImageTiffSplitter(tiff_path=tiff_path)
        for i_z, z_value in enumerate(z_value_list):
            i_roi = i_z//n_z_per_roi
            arr = splitter._get_pages(i_roi=i_roi,
                                      z_value=z_value)
            assert len(arr) == 5
            for i_page in range(5):
                expected = page_lookup[(i_roi, z_value)][i_page]
                np.testing.assert_array_equal(expected,
                                              arr[i_page])

            tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix='.tiff')[1]
            tmp_path = pathlib.Path(tmp_path)
            splitter.write_image_tiff(i_roi=i_roi,
                                      z_value=z_value,
                                      tiff_path=tmp_path)
            with tifffile.TiffFile(tmp_path, 'rb') as tiff_file:
                assert len(tiff_file.pages) == 1
                actual = tiff_file.pages[0].asarray()

            np.testing.assert_array_equal(actual,
                                          avg_img_lookup[(i_roi, z_value)])
            if tmp_path.is_file():
                tmp_path.unlink()
        if tiff_path.is_file():
            tiff_path.unlink()


@pytest.mark.parametrize(
    "z_value_list, n_rois, use_zs",
    [(list(range(8)), 2, True),
     (list(range(8)), 4, True),
     (list(range(6)), 3, True),
     ((0, 2, 3, 5, 7, 4, 6, 11), 4, True),
     ((0, 2, 3, 5, 7, 4, 6, 11), 2, True),
     ((0, 2, 3, 5, 7, 4, 6, 5), 4, True),
     ((0, 2, 3, 5, 7, 4, 6, 5), 2, True),
     ((0, 4, 5, 2, 8, 7), 1, True),
     (list(range(8)), 2, False),
     (list(range(8)), 4, False),
     (list(range(6)), 3, False),
     ((0, 2, 3, 5, 7, 4, 6, 11), 4, False),
     ((0, 2, 3, 5, 7, 4, 6, 11), 2, False),
     ((0, 2, 3, 5, 7, 4, 6, 5), 4, False),
     ((0, 2, 3, 5, 7, 4, 6, 5), 2, False),
     ((0, 4, 5, 2, 8, 7), 1, False)])
def test_time_splitter(tmp_path_factory,
                       z_value_list,
                       n_rois,
                       use_zs):
    n_rois = len(z_value_list)
    n_z_per_roi = 1

    tmp_dir = tmp_path_factory.mktemp('test_depth_2x4')

    time_tiff = _create_image_tiff(
                    pathlib.Path(tmp_dir),
                    z_value_list,
                    n_rois,
                    use_zs=use_zs,
                    is_surface=True)

    tiff_path = time_tiff[0]
    page_lookup = time_tiff[2]
    metadata = time_tiff[3]

    def scanimage_metadata_mock(file_handle):
        return metadata

    with patch('tifffile.read_scanimage_metadata',
               new=scanimage_metadata_mock):

        splitter = TimeSeriesSplitter(tiff_path=tiff_path)
        for i_z, z_value in enumerate(z_value_list):
            i_roi = i_z//n_z_per_roi
            tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix='.h5')[1]
            tmp_path = pathlib.Path(tmp_path)
            splitter.write_video_h5(
                        i_roi=i_roi,
                        z_value=z_value,
                        h5_path=tmp_path)
            with h5py.File(tmp_path, 'r') as in_file:
                actual = in_file['data'][()]
            expected = np.stack(page_lookup[(i_roi, z_value)])
            np.testing.assert_array_equal(expected, actual)

            if tmp_path.is_file():
                tmp_path.unlink()
        if tiff_path.is_file():
            tiff_path.unlink()


def _create_z_stack_tiffs(
        tmpdir: pathlib.Path,
        roi_to_z_mapping: List[List[int]],
        use_zs: bool = False):
    """
    Returns
    -------
    dict mapping z_stack_path to metadata
    dict mapping (i_roi, z_value) to tiff_pages_lookup
    """

    rng = np.random.default_rng()

    tiff_pages_lookup = dict()
    z_stack_path_to_metadata = dict()

    n_rois = len(roi_to_z_mapping)

    if use_zs:
        z_key = 'SI.hStackManager.zs'
    else:
        z_key = 'SI.hStackManager.zsAllActuators'

    main_roi_list = []
    for ii in range(n_rois):
        main_roi_list.append({'zs': -10,  # for z_stack, ;zs' does not matter
                              'discretePlaneMode': 1})

    main_roi_metadata = {'RoiGroups':
                         {'imagingRoiGroup':
                          {'rois': main_roi_list}}}

    n_repeats = 5

    for i_roi in range(n_rois):
        stack_path = pathlib.Path(tempfile.mkstemp(
                                       dir=tmpdir,
                                       suffix='.tiff')[1])
        roi_metadata = copy.deepcopy(main_roi_metadata)
        rois = roi_metadata['RoiGroups']['imagingRoiGroup']['rois']
        rois[i_roi]['discretePlaneMode'] = 0

        z0 = roi_to_z_mapping[i_roi][0]
        z0_values = np.linspace(z0-5.0, z0+5.0, 7)
        z1 = roi_to_z_mapping[i_roi][1]
        z1_values = np.linspace(z1-5.0, z1+5.0, 7)
        z_values = []
        for ii in range(len(z0_values)):
            z_values.append([z0_values[ii], z1_values[ii]])

        metadata = [{z_key: z_values},
                    roi_metadata]
        str_path = str(stack_path.resolve().absolute())
        z_stack_path_to_metadata[str_path] = metadata

        this_tiff = []
        for i_z in range(len(z0_values)):
            for i_repeat in range(n_repeats):
                for ii in (0, 1):
                    this_z = roi_to_z_mapping[i_roi][ii]
                    if (i_roi, this_z) not in tiff_pages_lookup:
                        tiff_pages_lookup[(i_roi, this_z)] = []
                    page = rng.integers(0, 2**16-1, (24, 24)).astype(np.int16)
                    page[i_roi:i_roi+5, i_roi:i_roi+5] = 0
                    page[i_z:i_z+2, i_z:i_z+2] = 1000
                    tiff_pages_lookup[(i_roi, this_z)].append(page)
                    this_tiff.append(page)

        tifffile.imsave(stack_path, this_tiff)

    return (z_stack_path_to_metadata,
            tiff_pages_lookup)


@pytest.mark.parametrize(
    "z_value_list, use_zs",
    [(list(range(8)), True),
     (list(range(8)), True),
     (list(range(6)), True),
     ((0, 2, 3, 5, 7, 4, 6, 11), True),
     ((0, 2, 3, 5, 7, 4, 6, 11), True),
     ((0, 2, 3, 5, 7, 4, 6, 5), True),
     ((0, 2, 3, 5, 7, 4, 6, 5), True),
     ((0, 4, 5, 2, 8, 7), True),
     (list(range(8)), False),
     (list(range(8)), False),
     (list(range(6)), False),
     ((0, 2, 3, 5, 7, 4, 6, 11), False),
     ((0, 2, 3, 5, 7, 4, 6, 11), False),
     ((0, 2, 3, 5, 7, 4, 6, 5), False),
     ((0, 2, 3, 5, 7, 4, 6, 5), False),
     ((0, 4, 5, 2, 8, 7), False)])
def test_z_stack_splitter(tmp_path_factory,
                          z_value_list,
                          use_zs):
    tmpdir = pathlib.Path(tmp_path_factory.mktemp('z_stack_test'))

    n_z_per_roi = 2  # a requirement of the z-stacks
    n_rois = len(z_value_list) // n_z_per_roi
    roi_to_z_mapping = []
    for i_roi in range(n_rois):
        i0 = i_roi*n_z_per_roi
        i1 = i0 + n_z_per_roi
        these_zs = z_value_list[i0:i1]
        roi_to_z_mapping.append(these_zs)

    dataset = _create_z_stack_tiffs(
                    tmpdir=tmpdir,
                    roi_to_z_mapping=roi_to_z_mapping,
                    use_zs=use_zs)

    z_stack_path_to_metadata = dataset[0]
    tiff_pages_lookup = dataset[1]

    def mock_read_metadata(tiff_path):
        str_path = str(tiff_path.resolve().absolute())
        return z_stack_path_to_metadata[str_path]

    to_replace = 'ophys_etl.modules.mesoscope_splitting.'
    to_replace += 'tiff_metadata._read_metadata'
    with patch(to_replace,
               new=mock_read_metadata):

        z_stack_path_list = [pathlib.Path(k)
                             for k in z_stack_path_to_metadata]
        splitter = ZStackSplitter(
                        tiff_path_list=z_stack_path_list)

        for i_roi in range(n_rois):
            for z_value in roi_to_z_mapping[i_roi]:
                actual = splitter._get_pages(
                                     i_roi=i_roi,
                                     z_value=z_value)
                expected = np.stack(tiff_pages_lookup[(i_roi, z_value)])
                np.testing.assert_array_equal(actual, expected)

                tmp_h5 = tempfile.mkstemp(dir=tmpdir, suffix='.h5')[1]
                tmp_h5 = pathlib.Path(tmp_h5)
                splitter.write_stack_h5(
                            i_roi=i_roi,
                            z_value=z_value,
                            zstack_path=tmp_h5)
                with h5py.File(tmp_h5, 'r') as in_file:
                    actual = in_file['data'][()]
                np.testing.assert_array_equal(actual, expected)

                if tmp_h5.is_file():
                    tmp_h5.unlink()
        for z_stack_path in z_stack_path_list:
            if z_stack_path.is_file():
                z_stack_path.unlink()
