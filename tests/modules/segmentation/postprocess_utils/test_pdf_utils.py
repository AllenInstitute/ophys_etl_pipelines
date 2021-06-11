import numpy as np

from ophys_etl.modules.segmentation.postprocess_utils.roi_merging import (
    make_cdf)

def test_cdf():
    rng = np.random.RandomState(221144)
    data_set_list = [rng.random_sample(200)*5.0,
                     rng.normal(5.0, 2.0, size=300)]
    for data_set in data_set_list:
        (bins,
         cdf) = make_cdf(data_set)
        test = rng.random_sample(50)*5.2
        interped_cdf = np.interp(test, bins, cdf)
        for xx, yy in zip(test, interped_cdf):
            ct = (data_set<xx).sum()
            assert np.abs(yy-ct/len(data_set)) < 1.0e-2

