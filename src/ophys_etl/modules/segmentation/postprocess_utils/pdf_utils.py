import numpy as np


def make_cdf(img_flat):
    val, val_ct = np.unique(img_flat, return_counts=True)
    cdf = np.cumsum(val_ct)
    cdf = cdf/val_ct.sum()
    assert len(val) == len(cdf)
    assert cdf.max()<=1.0
    assert cdf.min()>=0.0
    return val, cdf


def cdf_to_pdf(cdf_bins, cdf_vals):

    delta = np.diff(cdf_bins)
    pdf_bins = np.concatenate([np.array([cdf_bins[0]-0.5*delta[0]]),
                               cdf_bins,
                               np.array([cdf_bins[-1]+0.5*delta[-1]])])

    pdf_values = np.zeros(len(pdf_bins), dtype=float)

    pdf_values[1] = (cdf_vals[1]-cdf_vals[0])/(cdf_bins[1]-cdf_bins[0])
    pdf_values[-2] = (cdf_vals[-1]-cdf_vals[-2])/(cdf_bins[-1]-cdf_bins[-2])

    num = (cdf_vals[2:]-cdf_vals[:-2])
    denom = (cdf_bins[2:]-cdf_bins[:-2])
    pdf_values[2:-2] = num/denom

    return pdf_bins, pdf_values
