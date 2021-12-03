import pytest
from unittest.mock import MagicMock, Mock
from pathlib import Path
from functools import partial
import os
import numpy as np
import h5py
import json

import sys
sys.modules['suite2p'] = Mock()
from ophys_etl.modules.suite2p_wrapper \
        import __main__ as suite2p_wrapper  # noqa: E402
from ophys_etl.modules.suite2p_wrapper.utils \
        import Suite2PWrapperException  # noqa: E402

suite2p_basenames = ['ops1.npy', 'data.bin', 'Fneu.npy', 'F.npy', 'iscell.npy',
                     'ops.npy', 'spks.npy', 'stat.npy']


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
        movie_frame_rate_hz, exception):
    nframes = 100
    input_file = tmp_path / "input.h5py"
    with h5py.File(input_file, "w") as f:
        f.create_dataset('data', data=np.zeros((nframes, 34, 45)))

    args = {
            'h5py': str(input_file),
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
