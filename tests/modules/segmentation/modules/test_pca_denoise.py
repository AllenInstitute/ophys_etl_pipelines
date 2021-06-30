import pytest
import h5py
import numpy as np
from pathlib import Path

from ophys_etl.modules.segmentation.modules import pca_denoise


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


@pytest.mark.parametrize(
        "random_video_path",
        [{"video_shape": (100, 20, 20)}], indirect=True)
def test_PCADenoise(random_video_path, tmpdir):
    output = tmpdir / "out.h5"
    args = {
            "video_path": str(random_video_path),
            "video_output": str(output),
            "h5_chunk_shape": (50, 10, 10),
            "n_components": 10,
            "n_chunks": 2}
    sd = pca_denoise.PCADenoise(input_data=args, args=[])
    sd.run()
    with h5py.File(random_video_path, "r") as f:
        inshape = f["data"].shape
        intype = f["data"].dtype
    with h5py.File(output, "r") as f:
        outshape = f["data"].shape
        outtype = f["data"].dtype

    assert outshape == inshape
    assert intype == outtype
