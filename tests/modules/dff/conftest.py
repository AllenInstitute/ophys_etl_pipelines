import pytest
import h5py
import numpy as np


@pytest.fixture
def trace_h5(tmp_path):
    with h5py.File(tmp_path / "input_trace.h5", "w") as f:
        f.create_dataset("FC", data=np.ones((3, 100)))
        f.create_dataset("roi_names",
                         data=np.array([b'abc', b'123', b'drm'], dtype='|S3'))
    f.close()
    yield
