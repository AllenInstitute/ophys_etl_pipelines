import pytest
import numpy as np
import pandas as pd

from ophys_etl.modules.trace_extraction.utils import calculate_traces


@pytest.fixture
def video(image_dims):
    num_frames = 20
    data = np.ones((num_frames, image_dims['height'], image_dims['width']))
    data[:, 50:, 50:] = 2
    return data


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
