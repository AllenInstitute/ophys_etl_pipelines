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


def pdf_to_entropy(pdf_bins, pdf_vals):
    ln_pdf = np.where(pdf_vals>0.0,
                      np.log(pdf_vals),
                      0.0)
    p_lnp = pdf_vals*ln_pdf
    integral = 0.5*(pdf_bins[1:]-pdf_bins[:-1])*(p_lnp[1:]+p_lnp[:-1])
    if not np.isfinite(integral).all():
        raise RuntimeError("non finite value in entropy integral")
    entropy = integral.sum()
    if not np.isfinite(entropy):
        raise RuntimeError("entropy is not finite")
    return -1.0*integral.sum()
