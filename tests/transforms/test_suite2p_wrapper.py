import pytest
from unittest.mock import MagicMock, Mock
from pathlib import Path
from functools import partial
import os
import numpy as np
import h5py
import json
import filecmp
import random

import sys
sys.modules['suite2p'] = Mock()
from ophys_etl.transforms import suite2p_wrapper # noqa

suite2p_basenames = ['ops1.npy', 'data.bin', 'Fneu.npy', 'F.npy', 'iscell.npy',
                     'ops.npy', 'spks.npy', 'stat.npy']


@pytest.mark.parametrize(
        "basenames, dstsubdir, exception",
        [
            (['tmp.txt'], "other", False),
            (['tmp.txt', "tmp2.txt"], "other", False),
            (['tmp.txt', "tmp.txt"], "other", True),
            ])
def test_copy_and_add_uid(tmp_path, basenames, dstsubdir, exception):
    srcfiles = {}
    for i, bname in enumerate(basenames):
        if i == 0:
            tfile = tmp_path / bname
        else:
            spath = tmp_path / "sub_dir"
            spath.mkdir(exist_ok=True)
            tfile = spath / bname
        with open(tfile, "w") as f:
            f.write("%032x" % random.getrandbits(128))
        srcfiles[bname] = str(tfile)

    other_dir = tmp_path / "other"
    other_dir.mkdir()

    uid = "extra"
    if exception:
        with pytest.raises(ValueError, match=r".* Expected 1 match."):
            dstfiles = suite2p_wrapper.copy_and_add_uid(
                    tmp_path,
                    other_dir,
                    basenames,
                    uid)
    else:
        dstfiles = suite2p_wrapper.copy_and_add_uid(
                tmp_path,
                other_dir,
                basenames,
                uid)

        for bname in basenames:
            src = Path(srcfiles[bname])
            dst = Path(dstfiles[bname])
            assert filecmp.cmp(src, dst)
            assert dst.name == src.stem + f"_{uid}" + src.suffix


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


@pytest.mark.parametrize(
        "retain_files, nbinned, movie_frame_rate, exception",
        [
            ([['stat.npy'], 10, None, False]),
            ([['stat.npy'], None, 10, False]),
            ([['stat.npy'], None, None, True]),
            ([['stat.npy', 'F.npy'], None, None, True]),
            ([['all'], 10, None, False]),
            ])
def test_suite2p_wrapper(
        tmp_path, monkeypatch, retain_files, nbinned,
        movie_frame_rate, exception):
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
    if movie_frame_rate is not None:
        args['movie_frame_rate'] = movie_frame_rate

    mock_suite2p = MagicMock()
    mock_suite2p.run_s2p = MagicMock()
    mock_suite2p.run_s2p.side_effect = suite2p_side_effect

    mpatcher = partial(monkeypatch.setattr, target=suite2p_wrapper)
    mpatcher(name="suite2p", value=mock_suite2p)
    if exception:
        with pytest.raises(suite2p_wrapper.Suite2PWrapperException,
                           match=r".*`nbinned`.*"):
            s = suite2p_wrapper.Suite2PWrapper(input_data=args, args=[])
        return

    s = suite2p_wrapper.Suite2PWrapper(input_data=args, args=[])
    s.run()

    if movie_frame_rate is not None:
        bin_size = s.args['bin_duration'] * movie_frame_rate
        assert s.args['nbinned'] == int(nframes / bin_size)

    assert os.path.isfile(args['output_json'])
    with open(args['output_json'], 'r') as f:
        outj = json.load(f)

    file_list = retain_files
    if 'all' in retain_files:
        file_list = suite2p_basenames

    for fname in file_list:
        assert fname in list(outj['output_files'].keys())
        assert os.path.isfile(outj['output_files'][fname])
        assert s.now in outj['output_files'][fname]
