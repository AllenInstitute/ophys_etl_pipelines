import pytest
from unittest.mock import MagicMock, Mock
from pathlib import Path
from functools import partial
import os
import numpy as np
import h5py
import copy
import json

import sys
sys.modules['suite2p'] = Mock()
from ophys_etl.modules.suite2p_wrapper \
        import __main__ as suite2p_wrapper  # noqa: E402
from ophys_etl.modules.suite2p_wrapper.utils \
        import Suite2PWrapperException  # noqa: E402

suite2p_basenames = ['ops1.npy', 'data.bin', 'Fneu.npy', 'F.npy', 'iscell.npy',
                     'ops.npy', 'spks.npy', 'stat.npy']


@pytest.fixture(scope='session')
def input_movie_nframes_fixture():
    return 100


@pytest.fixture(scope='session')
def input_movie_path_fixture(tmpdir_factory,
                             input_movie_nframes_fixture):
    tmp_dir = Path(tmpdir_factory.mktemp('s2p_wrapper_video'))
    input_path = tmp_dir / "input.h5py"
    with h5py.File(input_path, "w") as f:
        f.create_dataset('data',
                         data=np.zeros((input_movie_nframes_fixture,
                                        34, 45)))
    yield str(input_path.resolve().absolute())


@pytest.fixture
def default_suite2p_args_fixture():
    this_dir = Path(__file__).parent
    resource_dir = this_dir / 'resources'
    file_path = resource_dir / 'default_s2p_args.json'
    with open(file_path, 'rb') as in_file:
        args = json.load(in_file)
    return args


@pytest.fixture
def allen_default_args_fixture():
    """
    Default Suite2P args set by our schema
    """
    output = dict()
    output['h5py_key'] = 'data'
    output['do_registration'] = 0
    output['reg_tif'] = False
    output['maxregshift'] = 0.2
    output['nimg_init'] = 200
    output['smooth_sigma'] = 1.15
    output['smooth_sigma_time'] = 4.0
    output['nonrigid'] = False
    output['roidetect'] = True
    output['sparse_mode'] = True
    output['diameter'] = 12
    output['spatial_scale'] = 0
    output['connected'] = True
    output['max_iterations'] = 20
    output['threshold_scaling'] = 0.75
    output['max_overlap'] = 0.75
    output['high_pass'] = 100
    output['smooth_masks'] = True
    output['inner_neuropil_radius'] = 2
    output['min_neuropil_pixels'] = 350
    output['allow_overlap'] = True
    return output


def suite2p_side_effect(args):
    """write the suite2p files at various depths
    """

    basedir = Path(args['save_path0'])
    nest1 = basedir / "nest1"
    nest2 = nest1 / "nest2"
    nest2.mkdir(parents=True)

    for bdir, flims in zip([basedir, nest1, nest2], [(0, 2), (2, 6), (6, 8)]):
        for i in range(flims[0], flims[1]):
            outfile = bdir / suite2p_basenames[i]
            with open(outfile, "w") as f:
                f.write('content')


@pytest.mark.suite2p_only
@pytest.mark.parametrize(
        "retain_files, nbinned, movie_frame_rate_hz, exception",
        [
            ([['stat.npy'], 10, None, False]),
            ([['stat.npy'], None, 10, False]),
            ([['stat.npy'], None, None, True]),
            ([['stat.npy', 'F.npy'], None, None, True]),
            ([['all'], 10, None, False]),
            ])
def test_suite2p_wrapper(
        tmp_path, monkeypatch, retain_files, nbinned,
        movie_frame_rate_hz, exception,
        input_movie_path_fixture,
        input_movie_nframes_fixture):

    args = {
            'h5py': input_movie_path_fixture,
            'output_dir': str(tmp_path / "output"),
            'output_json': str(tmp_path / "output.json"),
            'retain_files': retain_files
            }

    if nbinned is not None:
        args['nbinned'] = nbinned
    if movie_frame_rate_hz is not None:
        args['movie_frame_rate_hz'] = movie_frame_rate_hz

    mock_suite2p = MagicMock()
    mock_suite2p.run_s2p = MagicMock()
    mock_suite2p.run_s2p.side_effect = suite2p_side_effect

    def return_empty_dict():
        return dict()
    mock_suite2p.default_ops = return_empty_dict

    mpatcher = partial(monkeypatch.setattr, target=suite2p_wrapper)
    mpatcher(name="suite2p", value=mock_suite2p)
    if exception:
        with pytest.raises(Suite2PWrapperException,
                           match=r".*`nbinned`.*"):
            s = suite2p_wrapper.Suite2PWrapper(input_data=args, args=[])
        return

    s = suite2p_wrapper.Suite2PWrapper(input_data=args, args=[])
    s.run()

    if movie_frame_rate_hz is not None:
        bin_size = s.args['bin_duration'] * movie_frame_rate_hz
        nframes = input_movie_nframes_fixture
        assert s.args['nbinned'] == int(nframes / bin_size)

    assert os.path.isfile(args['output_json'])
    with open(args['output_json'], 'r') as f:
        outj = json.load(f)

    file_list = retain_files
    if 'all' in retain_files:
        file_list = suite2p_basenames

    for fname in file_list:
        assert fname in list(outj['output_files'].keys())
        assert os.path.isfile(outj['output_files'][fname][0])
        assert s.now in outj['output_files'][fname][0]


def compare_args(
        input_args,
        default_args,
        expected_changes):
    """
    input_args, default_args, and expected_changes are all dicts.
    This method will test that input_args is identical to
    default_args, except where expected_changes specifies an expected
    difference between the two.

    Note: input_args is allowed to contain extra keys that are not
    specified by default_args and expected_changes
    """

    msg = ''
    full_k_set = set(default_args.keys())
    full_k_set = full_k_set.union(set(expected_changes.keys()))
    for k in full_k_set:
        if k not in input_args:
            msg += f'missing key {k}\n'
    if len(msg) > 0:
        raise ValueError(msg)

    for k in input_args:
        if k not in full_k_set:
            continue
        if k == 'save_path0':
            if 'tmp_dir' in expected_changes:
                dirname = expected_changes['tmp_dir']
                if str(Path(input_args[k]).parent) != dirname:
                    msg += f'tmp_dir: {input_args[k]}; '
                    msg += f'parent should be {dirname}'
            continue

        if k in expected_changes:
            if input_args[k] != expected_changes[k]:
                msg += f'{k}: {input_args[k]}; '
                msg += 'expected_changes specifies '
                msg += f'{expected_changes[k]}\n'
        else:
            if input_args[k] != default_args[k]:
                msg += f'{k}: {input_args[k]}; '
                msg += 'default_args specifies '
                msg += f'{default_args[k]}\n'

    if len(msg) > 0:
        raise ValueError(msg)


def s2p_input_result_args(nframes):
    """
    This function is a generator which yields
    (by_hand_args, expected_args)

    by_hand_args ia s dict of non-default parameters that are to be
    passed into Suite2PWrapper

    expected_args is a dict of non-default parameters as we expect
    Suite2PWrapper to pass them into Suite2P

    Parameter
    ---------
    nframes: int
        The number of frames in the movie that Suite2P will be processing
    """
    possibilities = dict()
    possibilities['h5py_key'] = 'junk'
    possibilities['do_registration'] = 1
    possibilities['reg_tif'] = True
    possibilities['maxregshift'] = 0.5
    possibilities['nimg_init'] = 77
    possibilities['smooth_sigma'] = 0.83
    possibilities['smooth_sigma_time'] = 0.9
    possibilities['nonrigid'] = True
    possibilities['roidetect'] = False
    possibilities['sparse_mode'] = False
    possibilities['diameter'] = 66
    possibilities['spatial_scale'] = 8
    possibilities['connected'] = False
    possibilities['max_iterations'] = 11
    possibilities['threshold_scaling'] = 0.43
    possibilities['max_overlap'] = 0.11
    possibilities['high_pass'] = 92
    possibilities['smooth_masks'] = False
    possibilities['inner_neuropil_radius'] = 7
    possibilities['min_neuropil_pixels'] = 662
    possibilities['allow_overlap'] = False
    possibilities['movie_frame_rate_hz'] = 5.3

    k_list = list(possibilities.keys())
    k_list.sort()
    for leave_out in k_list:
        by_hand = dict()
        expected = dict()
        for k in possibilities:
            if k == leave_out:
                continue
            by_hand[k] = possibilities[k]
            expected[k] = possibilities[k]
            if k == 'movie_frame_rate_hz':
                expected['fs'] = possibilities[k]

        if 'movie_frame_rate_hz' not in by_hand:
            by_hand['nbinned'] = 22
            expected['nbinned'] = 22
        else:
            bin_size = 3.7*by_hand['movie_frame_rate_hz']
            nbinned = int(nframes/bin_size)
            by_hand['nbinned'] = nbinned
            expected['nbinned'] = nbinned

        yield (by_hand, expected)

    # now specify nbinned and movie_frame_rate_hz
    by_hand = copy.deepcopy(possibilities)
    expected = copy.deepcopy(possibilities)
    expected['fs'] = expected['movie_frame_rate_hz']
    by_hand['nbinned'] = 3141592654
    expected['nbinned'] = 3141592654
    yield (by_hand, expected)


def test_suite2p_default_args(tmp_path,
                              monkeypatch,
                              default_suite2p_args_fixture,
                              allen_default_args_fixture,
                              input_movie_path_fixture,
                              input_movie_nframes_fixture):
    """
    Test that Suite2P wrapper correctly combines the Suite2P default
    parameters with the user-specified parameters before sending
    arguments into Suite2P.

    Also test that the default Suite2P parameters that have not been
    overwritten are saved in Suite2PWrapper.args after run() is called.
    """
    mock_suite2p = MagicMock()
    mock_suite2p.run_s2p = MagicMock()
    mock_suite2p.run_s2p.side_effect = suite2p_side_effect

    def return_default_args():
        return copy.deepcopy(default_suite2p_args_fixture)
    mock_suite2p.default_ops = return_default_args

    mpatcher = partial(monkeypatch.setattr, target=suite2p_wrapper)
    mpatcher(name="suite2p", value=mock_suite2p)

    nframes = input_movie_nframes_fixture
    for by_hand_args, expected_args in s2p_input_result_args(nframes):
        args = dict()
        args['h5py'] = input_movie_path_fixture
        args['output_dir'] = str(tmp_path / 'output')
        args['output_json'] = str(tmp_path / 'output.json')
        for k in by_hand_args:
            args[k] = by_hand_args[k]

        expected_output = dict()
        expected_output['h5py'] = args['h5py']
        expected_output['output_dir'] = args['output_dir']
        expected_output['output_json'] = args['output_json']
        for k in allen_default_args_fixture:
            expected_output[k] = allen_default_args_fixture[k]

        for k in expected_args:
            expected_output[k] = expected_args[k]

        s = suite2p_wrapper.Suite2PWrapper(input_data=args, args=[])
        s.run()

        compare_args(s.args,
                     default_suite2p_args_fixture,
                     expected_output)
