import pytest
import numpy as np
import h5py
import tempfile
import pathlib
import hashlib
from itertools import product

from ophys_etl.utils.array_utils import (
    downsample_array,
    n_frames_from_hz)

from ophys_etl.modules.median_filtered_max_projection.utils import (
    apply_median_filter_to_video)

from ophys_etl.modules.downsample_video.utils import (
    _video_worker,
    create_downsampled_video_h5,
    _write_array_to_video,
    _min_max_from_h5,
    _video_array_from_h5,
    create_downsampled_video,
    create_side_by_side_video)


class DummyContextManager(object):
    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        return


@pytest.mark.parametrize(
    "output_hz, input_slice, kernel_size",
    product((12.0, 6.0, 4.0),
            ((6, 36), (12, 42), (30, 53)),
            (None, 2, 3)))
def test_ds_video_worker(
        tmpdir,
        ds_video_path_fixture,
        ds_video_array_fixture,
        output_hz,
        input_slice,
        kernel_size):
    """
    Test that _video_worker writes the expected result to the output file
    """

    input_hz = 12.0

    frames_to_group = n_frames_from_hz(
            input_hz,
            output_hz)

    # find the non-zero indices of the output file
    output_start = input_slice[0] // frames_to_group
    d_slice = input_slice[1] - input_slice[0]
    output_end = output_start + np.ceil(d_slice/frames_to_group).astype(int)

    output_path = tempfile.mkstemp(dir=tmpdir,
                                   prefix='ds_worker_test_',
                                   suffix='.h5')[1]
    output_path = pathlib.Path(output_path)

    with h5py.File(output_path, 'w') as out_file:
        dummy_data = np.zeros(ds_video_array_fixture.shape,
                              dtype=ds_video_array_fixture.dtype)
        out_file.create_dataset('data',
                                data=dummy_data)

    this_slice = ds_video_array_fixture[input_slice[0]:input_slice[1], :, :]

    if output_hz < input_hz:
        expected = downsample_array(this_slice,
                                    input_fps=input_hz,
                                    output_fps=output_hz,
                                    strategy='average')
    else:
        expected = np.copy(this_slice)

    if kernel_size is not None:
        expected = apply_median_filter_to_video(expected, kernel_size)

    lock = DummyContextManager()
    _video_worker(
            ds_video_path_fixture,
            input_hz,
            output_path,
            output_hz,
            kernel_size,
            input_slice,
            dict(),
            lock)

    with h5py.File(output_path, 'r') as in_file:
        full_data = in_file['data'][()]
    actual = full_data[output_start:output_end, :, :]
    np.testing.assert_array_equal(actual, expected)

    # make sure other pixels in output file were not touched
    other_mask = np.ones(full_data.shape, dtype=bool)
    other_mask[output_start:output_end, :, :] = False
    other_values = np.unique(full_data[other_mask])
    assert len(other_values) == 1
    assert np.abs(other_values[0]) < 1.0e-20


def test_ds_video_worker_exception(
        ds_video_path_fixture):
    """
    Test that exception is raised by _video_worker when input_slice[0]
    is not an integer multiple of the chunk size of frames used in
    downsampling
    """
    input_hz = 12.0
    output_hz = 6.0
    input_slice = [5, 19]
    kernel_size = 3
    output_path = pathlib.Path('silly.h5')

    with pytest.raises(RuntimeError, match="integer multiple"):
        lock = DummyContextManager()
        validity_dict = dict()
        _video_worker(
                ds_video_path_fixture,
                input_hz,
                output_path,
                output_hz,
                kernel_size,
                input_slice,
                validity_dict,
                lock)
    assert len(validity_dict) == 1
    for k in validity_dict:
        assert not validity_dict[k][0]
        assert "integer multiple" in validity_dict[k][1]


@pytest.mark.parametrize(
    "output_hz, kernel_size",
    product((12.0, 5.0, 3.0), (None, 2, 3)))
def test_create_ds_video_h5(
        tmpdir,
        ds_video_path_fixture,
        output_hz,
        kernel_size):
    """
    This is really just a smoke test
    """
    output_path = pathlib.Path(tempfile.mkstemp(
                                   dir=tmpdir,
                                   prefix="create_ds_vidoe_smoke_test_",
                                   suffix=".h5")[1])
    create_downsampled_video_h5(
        ds_video_path_fixture,
        12.0,
        output_path,
        output_hz,
        kernel_size,
        3)


@pytest.mark.parametrize(
    "video_suffix, fps, quality",
    product((".mp4", ".avi"),
            (5, 10),
            (3, 5, 8)))
def test_ds_write_array_to_video(
        tmpdir,
        ds_video_array_fixture,
        video_suffix,
        fps,
        quality):
    """
    This is just a smoke test of code that calls
    imageio to write the video files.
    """

    video_path = pathlib.Path(
                     tempfile.mkstemp(dir=tmpdir,
                                      prefix="dummy_",
                                      suffix=video_suffix)[1])

    _write_array_to_video(
        video_path,
        ds_video_array_fixture,
        fps,
        quality)

    assert video_path.is_file()


@pytest.mark.parametrize("border", (1, 2, 3, 100))
def test_min_max_from_h5_no_quantiles(
        ds_video_path_fixture,
        ds_video_array_fixture,
        border):

    nrows = ds_video_array_fixture.shape[1]
    ncols = ds_video_array_fixture.shape[2]

    if border < 8:
        this_array = ds_video_array_fixture[:,
                                            border:nrows-border,
                                            border:ncols-border]
    else:
        this_array = np.copy(ds_video_array_fixture)

    expected_min = this_array.min()
    expected_max = this_array.max()

    actual = _min_max_from_h5(
                    ds_video_path_fixture,
                    None,
                    border)

    assert np.abs(actual[0]-expected_min) < 1.0e-20
    assert np.abs(actual[1]-expected_max) < 1.0e-20


@pytest.mark.parametrize("border, quantiles",
                         product((1, 2, 3), ((0.1, 0.9), (0.2, 0.8))))
def test_min_max_from_h5_with_quantiles(
        ds_video_path_fixture,
        ds_video_array_fixture,
        border,
        quantiles):

    nrows = ds_video_array_fixture.shape[1]
    ncols = ds_video_array_fixture.shape[2]
    this_array = ds_video_array_fixture[:,
                                        border:nrows-border,
                                        border:ncols-border]

    (expected_min,
     expected_max) = np.quantile(this_array, quantiles)

    actual = _min_max_from_h5(
                    ds_video_path_fixture,
                    quantiles,
                    border)

    assert np.abs(actual[0]-expected_min) < 1.0e-20
    assert np.abs(actual[1]-expected_max) < 1.0e-20


@pytest.mark.parametrize(
    "min_val, max_val",
    product((50.0, 100.0, 250.0),
            (1900.0, 1500.0, 1000.0)))
def test_ds_video_array_from_h5_no_reticle(
        ds_video_path_fixture,
        ds_video_array_fixture,
        min_val,
        max_val):

    video_array = _video_array_from_h5(
                        ds_video_path_fixture,
                        min_val,
                        max_val,
                        reticle=False)

    assert video_array.dtype == np.uint8
    assert len(video_array.shape) == 4
    assert video_array.shape == (ds_video_array_fixture.shape[0],
                                 ds_video_array_fixture.shape[1],
                                 ds_video_array_fixture.shape[2],
                                 3)

    below_min = np.where(ds_video_array_fixture < min_val)
    assert len(below_min[0]) > 0
    assert (video_array[below_min] == 0).all()
    above_max = np.where(ds_video_array_fixture > max_val)
    assert len(above_max[0]) > 0
    assert (video_array[above_max] == 255).all()
    assert video_array.min() == 0
    assert video_array.max() == 255


@pytest.mark.parametrize("d_reticle", [5, 7, 9])
def test_ds_video_array_from_h5_with_reticle(
        ds_video_path_fixture,
        ds_video_array_fixture,
        d_reticle):

    min_val = 500.0
    max_val = 1500.0
    video_shape = ds_video_array_fixture.shape

    no_reticle = _video_array_from_h5(
                        ds_video_path_fixture,
                        min_val,
                        max_val,
                        reticle=False,
                        d_reticle=d_reticle)

    yes_reticle = _video_array_from_h5(
                        ds_video_path_fixture,
                        min_val,
                        max_val,
                        reticle=True,
                        d_reticle=d_reticle)

    reticle_mask = np.zeros(no_reticle.shape, dtype=bool)
    for ii in range(d_reticle, video_shape[1], d_reticle):
        reticle_mask[:, ii:ii+2, :, :] = True
    for ii in range(d_reticle, video_shape[2], d_reticle):
        reticle_mask[:, :, ii:ii+2, :] = True

    assert reticle_mask.sum() > 0

    np.testing.assert_array_equal(
            no_reticle[np.logical_not(reticle_mask)],
            yes_reticle[np.logical_not(reticle_mask)])

    assert not np.array_equal(no_reticle[reticle_mask],
                              yes_reticle[reticle_mask])


@pytest.mark.parametrize(
    "output_suffix, output_hz, kernel_size, quantiles, reticle, "
    "speed_up_factor, quality",
    product((".avi", ".mp4"),
            (3.0, 5.0),
            (2, 5),
            (None, (0.3, 0.9)),
            (True, False),
            (1, 4),
            (5, 7)))
def test_ds_create_downsampled_video(
        tmpdir,
        ds_video_path_fixture,
        ds_video_array_fixture,
        output_suffix,
        output_hz,
        kernel_size,
        quantiles,
        reticle,
        speed_up_factor,
        quality):
    """
    This will test create_downsampled_video by calling all of the
    constituent parts by hand and verifying that the md5checksum
    of the file produced that way matches the md5checksum of the file
    produced by calling create_downsampled_video. It's a little
    tautological, but it will tell us if any part of our pipeline is
    no longer self-consistent.
    """

    input_hz = 12.0
    d_reticle = 64  # because we haven't exposed this to the user, yet
    expected_file = pathlib.Path(
                        tempfile.mkstemp(dir=tmpdir,
                                         prefix='ds_expected_',
                                         suffix=output_suffix)[1])

    downsampled_video = downsample_array(
                            ds_video_array_fixture,
                            input_fps=input_hz,
                            output_fps=output_hz,
                            strategy='average')

    if kernel_size is not None:
        downsampled_video = apply_median_filter_to_video(
                                    downsampled_video,
                                    kernel_size)

    if quantiles is None:
        min_val = downsampled_video.min()
        max_val = downsampled_video.max()
    else:
        (min_val,
         max_val) = np.quantile(downsampled_video, quantiles)

    downsampled_video = downsampled_video.astype(float)
    downsampled_video = np.where(downsampled_video > min_val,
                                 downsampled_video, min_val)
    downsampled_video = np.where(downsampled_video < max_val,
                                 downsampled_video, max_val)

    delta = max_val-min_val
    downsampled_video = np.round(255.0*(downsampled_video-min_val)/delta)
    video_as_uint = np.zeros((downsampled_video.shape[0],
                              downsampled_video.shape[1],
                              downsampled_video.shape[2],
                              3), dtype=np.uint8)

    for ic in range(3):
        video_as_uint[:, :, :, ic] = downsampled_video

    video_shape = video_as_uint.shape
    del downsampled_video

    if reticle:
        for ii in range(d_reticle, video_shape[1], d_reticle):
            old_vals = np.copy(video_as_uint[:, ii:ii+2, :, :])
            new_vals = np.zeros(old_vals.shape, dtype=np.uint8)
            new_vals[:, :, :, 0] = 255
            new_vals = (new_vals//2) + (old_vals//2)
            new_vals = new_vals.astype(np.uint8)
            video_as_uint[:, ii:ii+2, :, :] = new_vals
        for ii in range(d_reticle, video_shape[2], d_reticle):
            old_vals = np.copy(video_as_uint[:, :, ii:ii+2, :])
            new_vals = np.zeros(old_vals.shape, dtype=np.uint8)
            new_vals[:, :, :, 0] = 255
            new_vals = (new_vals//2) + (old_vals//2)
            new_vals = new_vals.astype(np.uint8)
            video_as_uint[:, :, ii:ii+2, :] = new_vals

    _write_array_to_video(
            expected_file,
            video_as_uint,
            int(speed_up_factor*output_hz),
            quality)

    assert expected_file.is_file()

    actual_file = pathlib.Path(
                        tempfile.mkstemp(dir=tmpdir,
                                         prefix='ds_actual_',
                                         suffix=output_suffix)[1])

    create_downsampled_video(
            ds_video_path_fixture,
            input_hz,
            actual_file,
            output_hz,
            kernel_size,
            3,
            quality=quality,
            quantiles=quantiles,
            reticle=reticle,
            speed_up_factor=speed_up_factor,
            tmp_dir=tmpdir)

    assert actual_file.is_file()

    md5_expected = hashlib.md5()
    with open(expected_file, 'rb') as in_file:
        chunk = in_file.read(100000)
        while len(chunk) > 0:
            md5_expected.update(chunk)
            chunk = in_file.read(100000)

    md5_actual = hashlib.md5()
    with open(actual_file, 'rb') as in_file:
        chunk = in_file.read(100000)
        while len(chunk) > 0:
            md5_actual.update(chunk)
            chunk = in_file.read(100000)

    assert md5_actual.hexdigest() == md5_expected.hexdigest()


@pytest.mark.parametrize(
    "output_suffix, output_hz, kernel_size, quantiles, reticle, "
    "speed_up_factor, quality",
    product((".avi", ".mp4"),
            (3.0, 5.0),
            (2, 5),
            (None, (0.3, 0.9)),
            (True, False),
            (1, 4),
            (5, 7)))
def test_ds_create_side_by_side_video(
        tmpdir,
        ds_video_path_fixture,
        ds_video_array_fixture,
        output_suffix,
        output_hz,
        kernel_size,
        quantiles,
        reticle,
        speed_up_factor,
        quality):
    """
    This is just going to be a smoke test, as it's hard to verify
    the contents of an mp4
    """

    actual_file = pathlib.Path(
                        tempfile.mkstemp(dir=tmpdir,
                                         prefix='ds_side_by_side_actual_',
                                         suffix=output_suffix)[1])

    input_hz = 12.0
    create_side_by_side_video(
            ds_video_path_fixture,
            ds_video_path_fixture,
            input_hz,
            actual_file,
            output_hz,
            kernel_size,
            3,
            quality,
            quantiles,
            reticle,
            speed_up_factor,
            tmpdir)

    assert actual_file.is_file()
