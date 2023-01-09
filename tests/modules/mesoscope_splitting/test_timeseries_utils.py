import pytest
import pathlib
import h5py
import tifffile
import json
from itertools import product
import numpy as np
from ophys_etl.modules.mesoscope_splitting.timeseries_utils import (
    _dump_timeseries_caches,
    _gather_timeseries_caches,
    _split_timeseries_tiff,
    split_timeseries_tiff)


@pytest.fixture
def timeseries_fixtures():
    """
    A list of three random arrays to be passed
    around in caches and dumped to disk
    """
    rng = np.random.RandomState(1123)
    return [rng.random((37, 12, 12)),
            rng.random((37, 12, 12)),
            rng.random((36, 12, 12))]


@pytest.fixture
def timeseries_tiff_fixture(
        timeseries_fixtures,
        tmpdir_factory,
        helper_functions):
    """
    Write timeseries_fixtures into a TIFF
    """
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('tiff'))
    tiff_path = tmpdir / 'eg_timeseries.tiff'
    all_data = []
    for ii in range(37):
        for offset in range(3):
            if ii < timeseries_fixtures[offset].shape[0]:
                all_data.append(timeseries_fixtures[offset][ii, :, :])

    all_data = np.stack(all_data)
    tifffile.imsave(tiff_path, all_data)

    yield tiff_path

    helper_functions.clean_up_dir(tmpdir)


@pytest.mark.parametrize(
        "chunk_sizes", [(5, 11, 9),
                        (12, 30, 6),
                        (4, 18, 22),
                        (11, -1, 6)])
def test_dump_timeseries_caches(
        tmpdir_factory,
        helper_functions,
        timeseries_fixtures,
        chunk_sizes):
    """
    Test that dump_timeseries_caches writes the expected
    data to the specified files
    """

    tmpdir_list = [pathlib.Path(tmpdir_factory.mktemp('dump_1')),
                   pathlib.Path(tmpdir_factory.mktemp('dump_2')),
                   pathlib.Path(tmpdir_factory.mktemp('dump_3'))]
    offset_to_valid_cache = dict()
    offset_to_cache = dict()
    offset_to_tmp_files = dict()
    offset_to_tmp_dir = dict()
    for offset in range(3):
        cache = np.zeros((100,
                          timeseries_fixtures[offset].shape[1],
                          timeseries_fixtures[offset].shape[2]),
                         dtype=timeseries_fixtures[offset].dtype)

        if chunk_sizes[offset] > 0:
            chunk = timeseries_fixtures[offset][:chunk_sizes[offset],
                                                :, :]
            cache[:chunk_sizes[offset], :, :] = chunk

        offset_to_tmp_files[offset] = []
        offset_to_tmp_dir[offset] = tmpdir_list[offset]
        offset_to_valid_cache[offset] = chunk_sizes[offset]
        offset_to_cache[offset] = cache

    _dump_timeseries_caches(
        offset_to_cache=offset_to_cache,
        offset_to_valid_cache=offset_to_valid_cache,
        offset_to_tmp_files=offset_to_tmp_files,
        offset_to_tmp_dir=offset_to_tmp_dir)

    for offset in range(3):
        assert offset_to_valid_cache[offset] == -1
        if chunk_sizes[offset] > 0:
            assert len(offset_to_tmp_files[offset]) == 1
            filepath = offset_to_tmp_files[offset][0]
            with h5py.File(filepath, 'r') as in_file:
                expected = in_file['data'][()]
            actual = timeseries_fixtures[offset][:chunk_sizes[offset],
                                                 :, :]
            np.testing.assert_allclose(
                    expected, actual)
        else:
            assert len(offset_to_tmp_files[offset]) == 0

    for tmpdir in tmpdir_list:
        helper_functions.clean_up_dir(tmpdir)


@pytest.mark.parametrize(
        "expected_metadata",
        [{'a': 1, 'b': 2}, None])
def test_gather_timeseries_caches(
        tmpdir_factory,
        helper_functions,
        expected_metadata):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('gather'))
    finaldir = pathlib.Path(tmpdir_factory.mktemp('gather_output'))
    rng = np.random.default_rng(581321)

    data = rng.random((29, 13, 13))
    file_path_list = []
    for fname, i0, i1 in zip(('cache0.h5', 'cache1.h5', 'cache2.h5'),
                             (0, 13, 22),
                             (13, 22, 29)):
        file_path = tmpdir / fname
        file_path_list.append(file_path)
        with h5py.File(file_path, 'w') as out_file:
            out_file.create_dataset('data', data=data[i0:i1, :, :])

    full_path = finaldir / 'full.h5'

    _gather_timeseries_caches(
        file_path_list=file_path_list,
        final_output_path=full_path,
        metadata=expected_metadata)

    with h5py.File(full_path, 'r') as in_file:

        if expected_metadata is None:
            assert 'scanimage_metadata' not in in_file.keys()
        else:
            actual_metadata = json.loads(
                        in_file['scanimage_metadata'][()].decode('utf-8'))
            assert actual_metadata == expected_metadata

        actual = in_file['data'][()]

    np.testing.assert_allclose(data, actual)

    # make sure that tmpdir got cleaned up automatically
    filename_list = [f for f in tmpdir.iterdir()]
    assert len(filename_list) == 0

    helper_functions.clean_up_dir(tmpdir)
    helper_functions.clean_up_dir(finaldir)


@pytest.mark.parametrize(
        'dump_every, same_tmpdir, expected_metadata',
        product((50, 17, 8),
                (True, False),
                (None, {'a': 1, 'b': 2})))
def test_split_timeseries_tiff_worker(
        timeseries_fixtures,
        timeseries_tiff_fixture,
        tmpdir_factory,
        helper_functions,
        dump_every,
        same_tmpdir,
        expected_metadata):

    offset_to_tmp_dir = dict()
    if same_tmpdir:
        tmpdir = pathlib.Path(tmpdir_factory.mktemp('split_caches'))
        for offset in range(3):
            offset_to_tmp_dir[offset] = tmpdir
    else:
        for offset in range(3):
            tmpdir = pathlib.Path(
                    tmpdir_factory.mktemp(f'split_caches{offset}'))
            offset_to_tmp_dir[offset] = tmpdir

    final_dir = pathlib.Path(tmpdir_factory.mktemp('final_timeseries'))

    offset_to_tmp_files = dict()
    offset_to_path = dict()
    for offset in range(3):
        offset_to_tmp_files[offset] = []
        offset_to_path[offset] = final_dir / f'final{offset}.h5'

    _split_timeseries_tiff(
            tiff_path=timeseries_tiff_fixture,
            offset_to_path=offset_to_path,
            offset_to_tmp_files=offset_to_tmp_files,
            offset_to_tmp_dir=offset_to_tmp_dir,
            dump_every=dump_every,
            metadata=expected_metadata)

    for offset in range(3):
        expected = timeseries_fixtures[offset]
        with h5py.File(offset_to_path[offset], 'r') as in_file:
            if expected_metadata is None:
                assert 'scanimage_metadata' not in in_file.keys()
            else:
                actual_metadata = json.loads(
                          in_file['scanimage_metadata'][()].decode('utf-8'))
                assert actual_metadata == expected_metadata

            actual = in_file['data'][()]
        np.testing.assert_allclose(expected, actual)

    helper_functions.clean_up_dir(final_dir)
    for tmpdir in set(offset_to_tmp_dir.values()):
        helper_functions.clean_up_dir(tmpdir)


@pytest.mark.parametrize(
        'dump_every, expected_metadata',
        product((50, 17, 8),
                (None, {'a': 1, 'b': 2})))
def test_split_timeseries_tiff(
        timeseries_fixtures,
        timeseries_tiff_fixture,
        tmpdir_factory,
        helper_functions,
        dump_every,
        expected_metadata):
    """
    Effectively the same test as test_split_timeseries_tiff_worker,
    except that this also tests that the cache files get cleaned
    up automatically.
    """
    tmpdir = pathlib.Path(
            tmpdir_factory.mktemp('split_tiff_parent'))

    final_dir = pathlib.Path(tmpdir_factory.mktemp('final_timeseries'))

    offset_to_tmp_files = dict()
    offset_to_path = dict()
    for offset in range(3):
        offset_to_tmp_files[offset] = []
        offset_to_path[offset] = final_dir / f'final{offset}.h5'

    split_timeseries_tiff(
            tiff_path=timeseries_tiff_fixture,
            offset_to_path=offset_to_path,
            tmp_dir=tmpdir,
            dump_every=dump_every,
            metadata=expected_metadata)

    for offset in range(3):
        expected = timeseries_fixtures[offset]
        with h5py.File(offset_to_path[offset], 'r') as in_file:
            if expected_metadata is None:
                assert 'scanimage_metadata' not in in_file.keys()
            else:
                actual_metadata = json.loads(
                        in_file['scanimage_metadata'][()].decode('utf-8'))
                assert actual_metadata == expected_metadata

            actual = in_file['data'][()]
        np.testing.assert_allclose(expected, actual)

    cache_paths = [f for f in tmpdir.iterdir()]
    assert len(cache_paths) == 0

    helper_functions.clean_up_dir(final_dir)
    helper_functions.clean_up_dir(tmpdir)
