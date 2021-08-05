import pytest
import numpy as np
import h5py

from ophys_etl.modules.segmentation.detect.feature_vector_rois import (
    calculate_pearson_feature_vectors,
    PearsonFeatureROI)


def test_calculate_pearson_feature_vectors():
    """
    run smoke test on calculate_pearson_feature_vectors
    """
    rng = np.random.RandomState(491852)
    data = rng.random_sample((100, 20, 20))
    mu = np.mean(data, axis=0).astype(float)
    seed_pt = (15, 3)
    features = calculate_pearson_feature_vectors(
                                data,
                                mu,
                                seed_pt)

    assert features.shape == (400, 400)

    # check that, if there is a mask, the appropriate
    # pixels are ignored
    mask = np.zeros((20, 20), dtype=bool)
    mask[4:7, 11:] = True
    features = calculate_pearson_feature_vectors(
                                data,
                                mu,
                                seed_pt,
                                pixel_ignore=mask)
    assert features.shape == (373, 373)


def test_roi_growth(example_video):
    """
    Test that ROI can run end to end even in cases of
    extreme masking
    """

    with h5py.File(example_video, 'r') as in_file:
        video_data = in_file['data'][()]

    video_mu = np.mean(video_data, axis=0).astype(float)

    seed_pt = (12, 16)
    origin = (0, 0)
    mask = np.zeros((40, 40), dtype=bool)

    # fewer than ten valid points
    mask[:, :] = True
    mask[12, 16] = False
    mask[1:4, 2] = False
    roi = PearsonFeatureROI(seed_pt, origin, video_data,
                            video_mu, pixel_ignore=mask)
    roi.get_mask()

    # only seed is valid
    mask[:, :] = True
    mask[12, 16] = False
    roi = PearsonFeatureROI(seed_pt, origin, video_data,
                            video_mu, pixel_ignore=mask)
    roi.get_mask()

    # nothing is valid
    mask[:, :] = True
    msg = "Tried to create ROI with no valid pixels"
    with pytest.raises(RuntimeError, match=msg):
        _ = PearsonFeatureROI(seed_pt, origin, video_data,
                              video_mu, pixel_ignore=mask)
