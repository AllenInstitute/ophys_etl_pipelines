import pytest
import random
from pathlib import Path
import filecmp

from ophys_etl.modules.suite2p_wrapper.utils import copy_and_add_uid


@pytest.mark.suite2p_also
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
            dstfiles = copy_and_add_uid(
                    tmp_path,
                    other_dir,
                    basenames,
                    uid)
    else:
        dstfiles = copy_and_add_uid(
                tmp_path,
                other_dir,
                basenames,
                uid)

        for bname in basenames:
            src = Path(srcfiles[bname])
            dst = Path(dstfiles[bname][0])
            assert filecmp.cmp(src, dst)
            assert dst.name == src.stem + f"_{uid}" + src.suffix
