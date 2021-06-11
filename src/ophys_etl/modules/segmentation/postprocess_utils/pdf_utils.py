import numpy as np


def make_cdf(img_flat):
    val, val_ct = np.unique(img_flat, return_counts=True)
    cdf = np.cumsum(val_ct)
    cdf = cdf/val_ct.sum()
    assert len(val) == len(cdf)
    assert cdf.max()<=1.0
    assert cdf.min()>=0.0
    return val, cdf


def cdf_to_pdf(cdf_bins0, cdf_vals0):

    #pdf_bins = np.concatenate([np.array([cdf_bins[0]-0.5*delta[0]]),
    #                           cdf_bins,
    #                           np.array([cdf_bins[-1]+0.5*delta[-1]])])

    cdf_bins = np.linspace(cdf_bins0.min(), cdf_bins0.max(), 1000)
    cdf_vals = np.interp(cdf_bins, cdf_bins0,cdf_vals0, left=0.0, right=1.0)

    pdf_values = np.zeros(len(cdf_bins), dtype=float)

    dx = 4
    for i_center in range(len(pdf_values)):
        if i_center<2:
            i0 = 0
            i1 = i0+dx
        elif i_center>len(pdf_values)-1-dx:
            i0 = len(pdf_values)-1-dx
            i1 = i0+dx
        else:
            i0 = i_center-2
            i1 = i_center+2
        pdf_values[i_center] = (cdf_vals[i1]-cdf_vals[i0])/(cdf_bins[i1]-cdf_bins[i0])

    delta = np.diff(cdf_bins)

    pdf_bins = np.concatenate([np.array([cdf_bins[0]-0.5*delta[0]]),
                               cdf_bins,
                               np.array([cdf_bins[-1]+0.5*delta[-1]])])

    pdf_values = np.concatenate([np.array([0.0]),
                                 pdf_values,
                                 np.array([0.0])])

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
