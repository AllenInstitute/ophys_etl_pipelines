import pytest
import numpy as np
import h5py
import pathlib
import tempfile
from itertools import product

from ophys_etl.utils.video_utils import (
    scale_video_to_uint8,
    _read_and_scale_all_at_once,
    _read_and_scale_by_chunks,
    read_and_scale)


@pytest.fixture(scope='session')
def chunked_video_path(tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('chunked_video'))
    fname = tempfile.mkstemp(dir=tmpdir,
                             prefix='example_large_video_chunked_',
                             suffix='.h5')[1]
    rng = np.random.RandomState(22312)
    with h5py.File(fname, 'w') as out_file:
        dataset = out_file.create_dataset('data',
                                          (214, 10, 10),
                                          chunks=(100, 5, 5),
                                          dtype=np.uint16)
        for chunk in dataset.iter_chunks():
            arr = rng.randint(0, 65536,
                              (chunk[0].stop-chunk[0].start,
                               chunk[1].stop-chunk[1].start,
                               chunk[2].stop-chunk[2].start))
            dataset[chunk] = arr

    fname = pathlib.Path(fname)
    yield fname
    fname.unlink()
    tmpdir.rmdir()


@pytest.fixture(scope='session')
def unchunked_video_path(tmpdir_factory):
    tmpdir = pathlib.Path(tmpdir_factory.mktemp('unchunked_video'))
    fname = tempfile.mkstemp(dir=tmpdir,
                             prefix='example_large_video_unchunked_',
                             suffix='.h5')[1]
    rng = np.random.RandomState(714432)
    with h5py.File(fname, 'w') as out_file:
        data = rng.randint(0, 65536, size=(214, 10, 10)).astype(np.uint16)
        out_file.create_dataset('data',
                                data=data,
                                chunks=None,
                                dtype=np.uint16)

    fname = pathlib.Path(fname)
    yield fname
    fname.unlink()
    tmpdir.rmdir()


def test_scale_video():

    data = np.array([[1.0, 2.0, 3.0, 4.0],
                     [5.0, 6.0, 7.0, 8.0]])

    scaled = scale_video_to_uint8(data,
                                  0.0,
                                  8.0)
    assert scaled.dtype == np.uint8

    expected = np.array([[32, 64, 96, 128],
                         [159, 191, 223, 255]]).astype(np.uint8)

    np.testing.assert_array_equal(expected, scaled)

    scaled = scale_video_to_uint8(data,
                                  0.0,
                                  15.0)
    expected = np.array([[17, 34, 51, 68],
                         [85, 102, 119, 136]]).astype(np.uint8)

    np.testing.assert_array_equal(expected, scaled)

    scaled = scale_video_to_uint8(data,
                                  2.0,
                                  7.0)

    expected = np.array([[0, 0, 51, 102],
                         [153, 204, 255, 255]]).astype(np.uint8)

    np.testing.assert_array_equal(expected, scaled)

    with pytest.raises(RuntimeError, match="in scale_video_to_uint8"):
        _ = scale_video_to_uint8(data, 1.0, 0.0)


@pytest.mark.parametrize(
         'to_use, normalization, geometry',
         product(('chunked', 'unchunked'),
                 ({'quantiles': None, 'min_max': (10, 5000)},
                  {'quantiles': (0.1, 0.9), 'min_max': None},
                  {'quantiles': None, 'min_max': None},
                  {'quantiles': (0.1, 0.9), 'min_max': (10, 5000)}),
                 ({'origin': (0, 0), 'frame_shape': None},
                  {'origin': (5, 5), 'frame_shape': (3, 3)})))
def test_read_and_scale_all_at_once(chunked_video_path,
                                    unchunked_video_path,
                                    to_use,
                                    normalization,
                                    geometry):
    if to_use == 'chunked':
        video_path = chunked_video_path
    elif to_use == 'unchunked':
        video_path = unchunked_video_path
    else:
        raise RuntimeError(f'bad to_use value: {to_use}')

    if normalization['quantiles'] is None and normalization['min_max'] is None:
        with pytest.raises(RuntimeError,
                           match='must specify either quantiles'):

            actual = _read_and_scale_all_at_once(
                        video_path,
                        geometry['origin'],
                        geometry['frame_shape'],
                        quantiles=normalization['quantiles'],
                        min_max=normalization['min_max'])
        return

    elif (normalization['quantiles'] is not None
          and normalization['min_max'] is not None):

        with pytest.raises(RuntimeError,
                           match='cannot specify both quantiles'):

            actual = _read_and_scale_all_at_once(
                        video_path,
                        geometry['origin'],
                        geometry['frame_shape'],
                        quantiles=normalization['quantiles'],
                        min_max=normalization['min_max'])
        return

    with h5py.File(video_path, 'r') as in_file:
        full_data = in_file['data'][()]
        if normalization['quantiles'] is not None:
            min_max = np.quantile(full_data, normalization['quantiles'])
        else:
            min_max = normalization['min_max']

    if geometry['frame_shape'] is None:
        frame_shape = full_data.shape[1:3]
    else:
        frame_shape = geometry['frame_shape']

    r0 = geometry['origin'][0]
    r1 = r0+frame_shape[0]
    c0 = geometry['origin'][1]
    c1 = c0+frame_shape[1]
    full_data = full_data[:, r0:r1, c0:c1]
    expected = scale_video_to_uint8(full_data, min_max[0], min_max[1])

    actual = _read_and_scale_all_at_once(
                    video_path,
                    geometry['origin'],
                    frame_shape,
                    quantiles=normalization['quantiles'],
                    min_max=normalization['min_max'])

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
         'to_use, normalization, geometry',
         product(('chunked', 'unchunked'),
                 ({'quantiles': None, 'min_max': (10, 5000)},
                  {'quantiles': (0.1, 0.9), 'min_max': None},
                  {'quantiles': None, 'min_max': None},
                  {'quantiles': (0.1, 0.9), 'min_max': (10, 5000)}),
                 ({'origin': (0, 0), 'frame_shape': None},
                  {'origin': (5, 5), 'frame_shape': (3, 3)})))
def test_read_and_scale_by_chunks(chunked_video_path,
                                  unchunked_video_path,
                                  to_use,
                                  normalization,
                                  geometry):
    if to_use == 'chunked':
        video_path = chunked_video_path
    elif to_use == 'unchunked':
        video_path = unchunked_video_path

    if normalization['quantiles'] is None and normalization['min_max'] is None:
        with pytest.raises(RuntimeError,
                           match='must specify either quantiles'):

            actual = _read_and_scale_by_chunks(
                        video_path,
                        geometry['origin'],
                        geometry['frame_shape'],
                        quantiles=normalization['quantiles'],
                        min_max=normalization['min_max'])
        return

    elif (normalization['quantiles'] is not None
          and normalization['min_max'] is not None):

        with pytest.raises(RuntimeError,
                           match='cannot specify both quantiles'):

            actual = _read_and_scale_by_chunks(
                        video_path,
                        geometry['origin'],
                        geometry['frame_shape'],
                        quantiles=normalization['quantiles'],
                        min_max=normalization['min_max'])
        return

    with h5py.File(video_path, 'r') as in_file:
        full_data = in_file['data'][()]
        if normalization['quantiles'] is not None:
            min_max = np.quantile(full_data, normalization['quantiles'])
        else:
            min_max = normalization['min_max']

    if geometry['frame_shape'] is None:
        frame_shape = full_data.shape[1:3]
    else:
        frame_shape = geometry['frame_shape']

    r0 = geometry['origin'][0]
    r1 = r0+frame_shape[0]
    c0 = geometry['origin'][1]
    c1 = c0+frame_shape[1]
    full_data = full_data[:, r0:r1, c0:c1]
    expected = scale_video_to_uint8(full_data, min_max[0], min_max[1])

    actual = _read_and_scale_by_chunks(
                    video_path,
                    geometry['origin'],
                    frame_shape,
                    quantiles=normalization['quantiles'],
                    min_max=normalization['min_max'])

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
         'to_use, normalization, geometry',
         product(('chunked', 'unchunked'),
                 ({'quantiles': None, 'min_max': (10, 5000)},
                  {'quantiles': (0.1, 0.9), 'min_max': None},
                  {'quantiles': None, 'min_max': None},
                  {'quantiles': (0.1, 0.9), 'min_max': (10, 5000)}),
                 ({'origin': (0, 0), 'frame_shape': None},
                  {'origin': (5, 5), 'frame_shape': (3, 3)})))
def test_read_and_scale(chunked_video_path,
                        unchunked_video_path,
                        to_use,
                        normalization,
                        geometry):
    if to_use == 'chunked':
        video_path = chunked_video_path
    elif to_use == 'unchunked':
        video_path = unchunked_video_path
    else:
        raise RuntimeError(f'bad to_use value: {to_use}')

    if normalization['quantiles'] is None and normalization['min_max'] is None:
        with pytest.raises(RuntimeError,
                           match='must specify either quantiles'):

            actual = read_and_scale(
                        video_path,
                        geometry['origin'],
                        geometry['frame_shape'],
                        quantiles=normalization['quantiles'],
                        min_max=normalization['min_max'])
        return

    elif (normalization['quantiles'] is not None
          and normalization['min_max'] is not None):

        with pytest.raises(RuntimeError,
                           match='cannot specify both quantiles'):

            actual = read_and_scale(
                        video_path,
                        geometry['origin'],
                        geometry['frame_shape'],
                        quantiles=normalization['quantiles'],
                        min_max=normalization['min_max'])
        return

    with h5py.File(video_path, 'r') as in_file:
        full_data = in_file['data'][()]
        if normalization['quantiles'] is not None:
            min_max = np.quantile(full_data, normalization['quantiles'])
        else:
            min_max = normalization['min_max']

    if geometry['frame_shape'] is None:
        frame_shape = full_data.shape[1:3]
    else:
        frame_shape = geometry['frame_shape']

    r0 = geometry['origin'][0]
    r1 = r0+frame_shape[0]
    c0 = geometry['origin'][1]
    c1 = c0+frame_shape[1]
    full_data = full_data[:, r0:r1, c0:c1]
    expected = scale_video_to_uint8(full_data, min_max[0], min_max[1])

    actual = read_and_scale(
                    video_path,
                    geometry['origin'],
                    frame_shape,
                    quantiles=normalization['quantiles'],
                    min_max=normalization['min_max'])

    np.testing.assert_array_equal(actual, expected)
