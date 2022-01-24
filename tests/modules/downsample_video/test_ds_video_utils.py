import pytest
import numpy as np
import h5py
import tempfile
from itertools import product

from ophys_etl.utils.array_utils import (
    downsample_array,
    n_frames_from_hz)

from ophys_etl.modules.median_filtered_max_projection.utils import (
    apply_median_filter_to_video)

from ophys_etl.modules.downsample_video.utils import (
    _video_worker)


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
