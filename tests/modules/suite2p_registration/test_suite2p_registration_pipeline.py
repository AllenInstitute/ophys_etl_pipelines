import pytest
import h5py
import pathlib
import tempfile
from itertools import product
import numpy as np

has_suite2p = True
try:
    import suite2p.registration  # noqa: F401
except ImportError:
    # need to get around the fact that Suite2P may not be defined
    # in the test environment. These tests should all be marked with
    # pytest.mark.suite2p_only, which means they will only be run in
    # our CircleCI environments that contain Suite2P
    has_suite2p = False

if has_suite2p:
    from ophys_etl.modules.suite2p_registration.__main__ import (
        Suite2PRegistration)


@pytest.fixture(scope='session')
def video_path_fixture(tmpdir_factory):
    video_dir = pathlib.Path(tmpdir_factory.mktemp('motion_correction_video'))
    video_path = tempfile.mkstemp(dir=video_dir,
                                  prefix='video_',
                                  suffix='.h5')[1]

    video_path = pathlib.Path(video_path)

    rng = np.random.default_rng(111999)
    ntime = 50

    # nrows, ncols needs to be larger than 128, since
    # the default block size for nonrigid motion correction
    # is 128x128
    nrows = 160
    ncols = 160
    data = np.round(rng.normal(6.0, 3.0, (ntime, nrows, ncols)))
    data = data.astype(np.int16)

    for rr in (10, 20, 30):
        for cc in (7, 12, 25, 29):
            amp = rng.random()*5.0+4.0
            r0 = max(0, rr-3)
            r1 = min(nrows, rr+3)
            c0 = max(0, cc-3)
            c1 = min(ncols, cc+3)
            roi = np.round(rng.normal(amp, 1.0, (ntime, r1-r0, c1-c0)))
            roi = roi.astype(np.int16)
            data[:, r0:r1, c0:c1] += roi

    for ii in range(ntime):
        dx = rng.integers(-5, 5)
        dy = rng.integers(-5, 5)
        data[ii, :, :] = np.roll(data[ii, :, :], (dx, dy), axis=(1, 0))

    with h5py.File(video_path, 'w') as out_file:
        out_file.create_dataset('data', data=data)
    yield video_path
    video_path.unlink()


@pytest.mark.suite2p_only
@pytest.mark.parametrize(
        "nonrigid, clip_negative",
        product((True, False), (True, False)))
def test_suite2p_motion_correction(
        tmpdir,
        nonrigid,
        clip_negative,
        video_path_fixture):

    corr_video_path = tempfile.mkstemp(
                            dir=tmpdir,
                            prefix='motion_corrected_',
                            suffix='.h5')[1]

    diagnostics_path = tempfile.mkstemp(
                            dir=tmpdir,
                            prefix='motion_diagnostics_',
                            suffix='.csv')[1]

    max_projection_path = tempfile.mkstemp(
                            dir=tmpdir,
                            prefix='max_projection_',
                            suffix='.png')[1]

    avg_projection_path = tempfile.mkstemp(
                            dir=tmpdir,
                            prefix='avg_projection_',
                            suffix='.png')[1]

    summary_path = tempfile.mkstemp(
                            dir=tmpdir,
                            prefix='summary_',
                            suffix='.png')[1]

    webm_path = tempfile.mkstemp(
                            dir=tmpdir,
                            prefix='preview_',
                            suffix='.webm')[1]

    output_json = tempfile.mkstemp(dir=tmpdir,
                                   prefix='output_',
                                   suffix='.json')[1]

    s2p_args = {'nonrigid': nonrigid,
                'h5py': str(video_path_fixture.resolve().absolute())}

    args = {'suite2p_args': s2p_args,
            'movie_frame_rate_hz': 6.1,
            'clip_negative': clip_negative,
            'motion_corrected_output': corr_video_path,
            'motion_diagnostics_output': diagnostics_path,
            'max_projection_output': max_projection_path,
            'avg_projection_output': avg_projection_path,
            'registration_summary_output': summary_path,
            'motion_correction_preview_output': webm_path,
            'output_json': output_json}

    runner = Suite2PRegistration(args=[],
                                 input_data=args)
    runner.run()

    with h5py.File(corr_video_path, 'r') as in_file:
        corrected_video = in_file['data'][()]

    # check that negative pixels were clipped correctly (or not)
    eps = 1.0e-10
    if clip_negative:
        assert corrected_video.min() >= 0.0
    else:
        assert corrected_video.min() < -1.0*eps

    with h5py.File(video_path_fixture, 'r') as in_file:
        input_video = in_file['data'][()]

    # check that rigid/nonrigid motion correction was applied as requested
    # (this is done by ordering the pixels in each frame and making sure,
    # for the case of rigid motion correction, that the sorted list of pixels
    # did not change). The rigid motion correction should preserve all pixel
    # values or all frames. nonrigid motion correction does not preserve all
    # pixel values and thus some frames should have different pixel values.
    n_non_rigid_different = 0
    for ii in range(input_video.shape[0]):
        in_pixels = np.sort(input_video[ii, :, :].flatten())
        out_pixels = np.sort(corrected_video[ii, :, :].flatten())
        if clip_negative:
            in_pixels = np.where(in_pixels > 0.0, in_pixels, 0.0)
        if nonrigid:
            if not np.array_equal(in_pixels, out_pixels):
                n_non_rigid_different += 1
        else:
            np.testing.assert_array_equal(in_pixels, out_pixels)
    if nonrigid:
        assert n_non_rigid_different > 0
