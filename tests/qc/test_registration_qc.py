import pytest
import h5py
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
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


@pytest.fixture
def videos(tmp_path):
    myarr = np.random.randint(0, 1000, size=(100, 10, 10), dtype='uint16')
    mypath1 = tmp_path / "video1.h5"
    with h5py.File(mypath1, "w") as f:
        f.create_dataset("data", data=myarr, chunks=(1, *myarr.shape[1:]))
    mypath2 = tmp_path / "video2.h5"
    with h5py.File(mypath2, "w") as f:
        f.create_dataset("data", data=myarr, chunks=(1, *myarr.shape[1:]))
    yield mypath1, mypath2


@pytest.fixture
def images(tmp_path):
    myarr = np.random.randint(0, 255, size=(10, 10), dtype='uint8')
    mypath1 = tmp_path / "image1.png"
    with Image.fromarray(myarr) as im:
        im.save(mypath1)
    mypath2 = tmp_path / "image2.png"
    with Image.fromarray(myarr) as im:
        im.save(mypath2)
    yield mypath1, mypath2


def test_registration_qc(tmp_path, images, videos, rigid_motion_csv):
    """
    """
    args = {
            "motion_diagnostics_output": rigid_motion_csv,
            "movie_frame_rate_hz": 11.0,
            "uncorrected_path": str(videos[0]),
            "motion_corrected_output": str(videos[1]),
            "max_projection_output": str(images[0]),
            "avg_projection_output": str(images[1]),
            "registration_summary_output": str(tmp_path / "summary.png"),
            "motion_correction_preview_output": str(tmp_path / "preview.webm")}

    reg = rqc.RegistrationQC(input_data=args, args=[])
    reg.run()

    for k in ['registration_summary_output',
              'motion_correction_preview_output']:
        assert Path(reg.args[k]).exists
