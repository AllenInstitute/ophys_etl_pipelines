import pytest
import numpy as np
import h5py
import pathlib
import tempfile
from typing import Union
from itertools import product

from ophys_etl.utils.video_utils import (
    _read_and_scale_all_at_once,
    _read_and_scale_by_chunks,
    read_and_scale)


def scale_video_to_uint8(video: np.ndarray,
                         min_value: Union[int, float],
                         max_value: Union[int, float]) -> np.ndarray:
    """
    Convert a video (as a numpy.ndarray) to uint8 by dividing by the
    array's maximum value and multiplying by 255

    Parameters
    ----------
    video: np.ndarray

    min_value: Optional[Union[int, float]]

    max_value: Optional[Union[int, float]]
        Video will be clipped at min_value and max_value and
        then normalized to (max_value-min_value) before being
        converted to uint8

    Returns
    -------
    np.ndarray

    Raises
    ------
    RuntimeError
        If min_value > max_value

    Notes
    -----
    Duplicate implementation of normalize_array taken from
    staging/segmentation_dev branch. Copying it here just for
    testing purposes.
    """

    if min_value > max_value:
        raise RuntimeError("in scale_video_to_uint8 "
                           f"min_value ({min_value}) > "
                           f"max_value ({max_value})")

    mask = video > max_value
    video[mask] = max_value
    mask = video < min_value
    video[mask] = min_value

    delta = (max_value-min_value)
    video = video-min_value
    return np.round(255*video.astype(float)/delta).astype(np.uint8)


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
