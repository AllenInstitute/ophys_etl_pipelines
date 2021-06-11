import numpy as np

from ophys_etl.modules.segmentation.postprocess_utils.pdf_utils import (
    make_cdf,
    cdf_to_pdf)

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


def test_cdf_to_pdf():
    rng = np.random.RandomState(1235412)
    data_set_list = [rng.random_sample(200)*5.0,
                     rng.normal(5.0, 2.0, size=300),
                     rng.chisquare(5, size=200)]
    for data_set in data_set_list:
        (cdf_bins,
         cdf_vals) = make_cdf(data_set)

        (pdf_bins,
         pdf_vals) = cdf_to_pdf(cdf_bins, cdf_vals)

        integral = 0.5*(pdf_bins[1:]-pdf_bins[:-1])*(pdf_vals[1:]+pdf_vals[:-1])
        integral = integral.sum()
        assert np.abs(1.0-integral) < 0.003

        assert pdf_vals[0] == 0.0
        assert pdf_vals[-1] == 0.0

        xmin = data_set.min()
        xmax = data_set.max()
        dx = xmax-xmin
        sample = xmin+rng.random_sample(size=200)*1.1*dx
        for x in sample:
            truth = np.interp(x,cdf_bins,cdf_vals,left=0.0,right=1.0)
            mask = (pdf_bins<=x)
            int_x = pdf_bins[mask]
            int_y = pdf_vals[mask]
            integral = 0.5*(int_x[1:]-int_x[:-1])*(int_y[:-1]+int_y[1:])
            integral = integral.sum()
            assert np.abs(integral-truth)<0.015
