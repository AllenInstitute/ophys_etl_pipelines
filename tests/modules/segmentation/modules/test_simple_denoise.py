import pytest
import h5py
import numpy as np
from pathlib import Path

from ophys_etl.modules.segmentation.modules import simple_denoise


@pytest.fixture
def random_video_path(tmpdir, request):
    # random values, used to smoke test functions
    vpath = Path(tmpdir / "video.h5")
    data = np.random.randint(0, 2**15,
                             size=request.param["video_shape"],
                             dtype='uint16')
    with h5py.File(vpath, "w") as f:
        f.create_dataset("data", data=data)
    yield vpath


@pytest.mark.parametrize("filter_type", ["uniform", "gaussian"])
@pytest.mark.parametrize("n_parallel_workers", [1, 2])
@pytest.mark.parametrize(
        "random_video_path",
        [{"video_shape": (100, 20, 20)}], indirect=True)
def test_SimpleDenoise(random_video_path, filter_type,
                       n_parallel_workers, tmpdir):
    output = tmpdir / "out.h5"
    args = {
            "video_path": str(random_video_path),
            "size": 10,
            "filter_type": filter_type,
            "n_parallel_workers": n_parallel_workers,
            "h5_chunk_shape": (50, 10, 10),
            "video_output": str(output)}
    sd = simple_denoise.SimpleDenoise(input_data=args, args=[])
    sd.run()
    with h5py.File(random_video_path, "r") as f:
        inshape = f["data"].shape
        intype = f["data"].dtype
    with h5py.File(output, "r") as f:
        outshape = f["data"].shape
        outtype = f["data"].dtype

    assert outshape == inshape
    assert intype == outtype
