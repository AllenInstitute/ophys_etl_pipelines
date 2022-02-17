import pytest
import random
from pathlib import Path
import hashlib

from ophys_etl.modules.suite2p_wrapper.utils import copy_and_add_uid


def path_to_hash(file_path):
    """
    Return the MD5 checksum of the file at file_path
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as in_file:
        chunk = in_file.read(10000)
        while len(chunk) > 0:
            hasher.update(chunk)
            chunk = in_file.read(10000)
    return str(hasher.hexdigest())


@pytest.mark.parametrize(
        "basenames, dstsubdir, use_mv, exception",
        [
            (['tmp.txt'], "other", False, False),
            (['tmp.txt', "tmp2.txt"], "other", False, False),
            (['tmp.txt', "tmp.txt"], "other", False, True),
            (['tmp.txt'], "other", True, False),
            (['tmp.txt', "tmp2.txt"], "other", True, False),
            (['tmp.txt', "tmp.txt"], "other", True, True),

            ])
def test_copy_and_add_uid(tmp_path,
                          basenames,
                          dstsubdir,
                          use_mv,
                          exception):
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
            dstfiles = copy_and_add_uid(
                    tmp_path,
                    other_dir,
                    basenames,
                    uid,
                    use_mv=use_mv)
    else:

        src_hash = dict()
        for bname in basenames:
            src_hash[bname] = path_to_hash(srcfiles[bname])

        dstfiles = copy_and_add_uid(
                tmp_path,
                other_dir,
                basenames,
                uid,
                use_mv=use_mv)

        for bname in basenames:
            src = Path(srcfiles[bname])
            dst = Path(dstfiles[bname][0])
            dst_hash = path_to_hash(dst)
            assert dst_hash == src_hash[bname]
            assert dst.name == src.stem + f"_{uid}" + src.suffix
            if use_mv:
                assert not src.exists()
            else:
                assert src.is_file()
