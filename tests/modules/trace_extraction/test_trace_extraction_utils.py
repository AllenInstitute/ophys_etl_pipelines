import pytest
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

from ophys_etl.modules.trace_extraction.utils import (
        calculate_traces,
        calculate_roi_and_neuropil_traces,
        extract_traces)


# image_dims fixture from tests/conftest.py
@pytest.fixture
def video(image_dims):
    num_frames = 20
    data = np.ones((num_frames, image_dims['height'], image_dims['width']))
    data[:, 50:, 50:] = 2
    return data


@pytest.fixture
def h5video(video, tmpdir):
    fname = tmpdir / "h5file.h5"
    with h5py.File(fname, "w") as f:
        f.create_dataset("data", data=video)
    yield str(fname)


# roi_mask_list fixture from tests/conftest.py
def test_calculate_traces(video, roi_mask_list):
    roi_traces, exclusions = calculate_traces(video, roi_mask_list)

    expected_exclusions = pd.DataFrame({
        'roi_id': ['0', '9'],
        'exclusion_label_name': ['motion_border', 'motion_border']
    })

    assert np.all(np.isnan(roi_traces[0, :]))
    assert np.all(roi_traces[4, :] == 1)
    assert np.all(roi_traces[6, :] == 2)
    assert np.all(np.isnan(roi_traces[9, :]))

    pd.testing.assert_frame_equal(expected_exclusions,
                                  pd.DataFrame(exclusions),
                                  check_like=True)


# roi_mask_list, motion_border fixture from tests/conftest.py
def test_calculate_roi_and_neuropil_traces(
        h5video, roi_mask_list, motion_border):
    roi_traces, neuropil_traces, neuropil_masks, exclusions = \
            calculate_roi_and_neuropil_traces(h5video,
                                              roi_mask_list,
                                              motion_border)

    assert neuropil_traces.shape == roi_traces.shape
    assert np.all(np.isnan(roi_traces[0, :]))
    assert np.all(roi_traces[4, :] == 1)
    assert np.all(roi_traces[6, :] == 2)
    assert np.all(np.isnan(roi_traces[9, :]))
    expected_exclusions = pd.DataFrame({
        'roi_id': ['0', '9'],
        'exclusion_label_name': ['motion_border', 'motion_border']
    })
    pd.testing.assert_frame_equal(expected_exclusions,
                                  pd.DataFrame(exclusions),
                                  check_like=True)


# motion_border_dict fixture from tests/conftest.py
# roi_list_of_dicts from tests/modules/trace_extraction/conftest.py
def test_extract_traces(h5video, roi_list_of_dicts,
                        motion_border_dict, tmpdir):
    sdir = Path(tmpdir)
    output = extract_traces(h5video,
                            motion_border_dict,
                            sdir,
                            roi_list_of_dicts)
    assert "exclusion_labels" in output
    assert isinstance(output["exclusion_labels"], list)
    expected_exclusions = pd.DataFrame({
        'roi_id': ['0', '9'],
        'exclusion_label_name': ['motion_border', 'motion_border']
    })
    pd.testing.assert_frame_equal(expected_exclusions,
                                  pd.DataFrame(output["exclusion_labels"]),
                                  check_like=True)

    for k in ["neuropil_trace_file", "roi_trace_file"]:
        assert k in output
        assert Path(output[k]).exists()
        assert Path(output[k]).is_file()
        assert Path(output[k]).parent == sdir

    with h5py.File(output["neuropil_trace_file"], "r") as f:
        n_names = f["roi_names"][()]
        n_traces = f["data"][()]
    with h5py.File(output["roi_trace_file"], "r") as f:
        r_names = f["roi_names"][()]
        r_traces = f["data"][()]
    np.testing.assert_array_equal(n_names, r_names)
    assert n_traces.shape == r_traces.shape
