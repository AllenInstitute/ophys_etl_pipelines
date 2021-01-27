import pytest
import h5py
import numpy as np
import pandas as pd
import ophys_etl.qc.registration_qc as rqc


@pytest.fixture
def rigid_motion_csv(tmp_path):
    df = pd.DataFrame({
        "framenumber": [0, 1, 2, 3],
        "x": [1, 2, 1, -1],
        "y": [2, 3, -2, 1],
        "correlation": [0.01, 0.02, 0.02, 0.03]})
    df_path = tmp_path / "rigid_motion.csv"
    df.to_csv(df_path)
    yield str(df_path)


def test_registration_qc_schema(tmp_path, rigid_motion_csv):
    """data via path or in-memory pass
    """
    myarr = np.random.randint(0, 1000, size=(100, 10, 10), dtype='uint16')
    args = {"motion_diagnostics_output": rigid_motion_csv}

    mypath = tmp_path / "motion_corr.h5"
    with h5py.File(mypath, "w") as f:
        f.create_dataset("data", data=myarr, chunks=(1, *myarr.shape[1:]))
    args["motion_corrected_path"] = str(mypath)
