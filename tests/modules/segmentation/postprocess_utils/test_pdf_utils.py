import numpy as np

from ophys_etl.modules.segmentation.postprocess_utils.pdf_utils import (
    make_cdf,
    cdf_to_pdf,
    pdf_to_entropy,
    _fit_with_wgt)

def test_fit_with_wgt():
    rng = np.random.RandomState(7142)
    xx = rng.random_sample(14)*50.0
    yy = 4.0*xx**2+15.0*xx+23.0+33.0*(rng.random_sample(14)-0.5)
    dx = 2.4
    for x0 in xx:
        w = np.exp(-0.5*((x0-xx)/dx)**2)
        m, b = _fit_with_wgt(x0, xx, yy, dx)
        dedb = np.sum(w*(yy-m*xx-b))
        dedm = np.sum(w*xx*(yy-m*xx-b))
        #print(dedm,dedb,m,b,w.min(),w.max())
        #assert np.abs(np.sum(w*b)-np.sum(w*(yy-m*xx)))<0.0001
        assert np.abs(dedb)<0.0001
        assert np.abs(dedm)<0.01


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
                     rng.chisquare(5, size=10)]
    for i_data, data_set in enumerate(data_set_list):
        (cdf_bins,
         cdf_vals) = make_cdf(data_set)

        (pdf_bins,
         pdf_vals) = cdf_to_pdf(cdf_bins, cdf_vals)

        integral = 0.5*(pdf_bins[1:]-pdf_bins[:-1])*(pdf_vals[1:]+pdf_vals[:-1])
        integral = integral.sum()
        assert np.abs(1.0-integral) < 0.003

        assert pdf_vals[0] == 0.0
        assert pdf_vals[-1] == 0.0
        assert pdf_vals[1:-1].min() >= 0.0

t        test_cdf = 0.5*(pdf_bins[1:]-pdf_bins[:-1])*(pdf_vals[1:]+pdf_vals[:-1])
        test_cdf = np.cumsum(test_cdf)
        #fig = figure.Figure(figsize=(10,10))
        #a = fig.add_subplot(1,1,1)
        #a.plot(cdf_bins, cdf_vals, color='b')
        #a.plot(pdf_bins[:-1],test_cdf, color='r')
        #a.plot(pdf_bins,pdf_vals/pdf_vals.max(), color='g')
        #fig.savefig(f'cdf_{i_data}.png')

        #xmin = data_set.min()
        #xmax = data_set.max()
        #dx = xmax-xmin
        #sample = xmin+rng.random_sample(size=200)*1.1*dx
        #for x in sample:
        #    truth = np.interp(x,cdf_bins,cdf_vals,left=0.0,right=1.0)
        #    mask = (pdf_bins<=x)
        #    int_x = pdf_bins[mask]
        #    int_y = pdf_vals[mask]
        #    integral = 0.5*(int_x[1:]-int_x[:-1])*(int_y[:-1]+int_y[1:])
        #    integral = integral.sum()
        #    #print(integral, truth, integral-truth)
        #    assert np.abs(integral-truth)<0.015

def test_pdf_to_entropy():
    rng = np.random.RandomState(427123)
    for i_iter in range(10):
        mu = 35.0+rng.random_sample()*40.0
        sigma = rng.random_sample()*10.0+0.01
        xx = np.linspace(mu-5*sigma, mu+5*sigma, 200)
        yy = np.exp(-0.5*((xx-mu)/sigma)**2)/np.sqrt(2.0*np.pi*sigma**2)

        check = np.sum(0.5*(xx[1:]-xx[:-1])*(yy[1:]+yy[:-1]))
        assert np.abs(check-1.0) < 0.002

        entropy = pdf_to_entropy(xx, yy)
        truth = np.log(sigma*np.sqrt(2.0*np.pi*np.exp(1)))
        assert np.abs(truth-entropy) < 0.001

        sigma *= 2.0

        data = rng.normal(mu, sigma, size=2000)
        (cdf_x,
         cdf_y) = make_cdf(data)
        (pdf_x,
         pdf_y) = cdf_to_pdf(cdf_x, cdf_y)
        xx = np.linspace(mu-5*sigma, mu+5*sigma, 200)
        yy = np.exp(-0.5*((xx-mu)/sigma)**2)/np.sqrt(2.0*np.pi*sigma**2)
        #fig = figure.Figure(figsize=(10,10))
        #ax = fig.add_subplot(1,1,1)
        #ax.plot(xx,yy,color='b',zorder=3)
        #ax.plot(pdf_x,pdf_y,color='r')
        #ax.plot(cdf_x,cdf_y,color='g')
        #fig.savefig(f'junk_{i_iter}.png')


        entropy = pdf_to_entropy(pdf_x, pdf_y)
        truth = np.log(sigma*np.sqrt(2.0*np.pi*np.exp(1)))
        assert np.abs(truth-entropy) < 0.1