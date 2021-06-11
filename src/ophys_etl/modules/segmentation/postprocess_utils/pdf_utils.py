import numpy as np
import time


def make_cdf(img_flat):
    val, val_ct = np.unique(img_flat, return_counts=True)
    cdf = np.cumsum(val_ct)
    cdf = cdf/val_ct.sum()
    assert len(val) == len(cdf)
    assert cdf.max()<=1.0
    assert cdf.min()>=0.0
    return val, cdf


def _fit_with_wgt(x0, xx, yy, dx):
    w = np.exp(-0.5*((x0-xx)/dx)**2)
    w = w/w.max()
    w_sum = w.sum()
    wy_sum = np.sum(w*yy)
    wxy_sum = np.sum(w*yy*xx)
    wx_sum = np.sum(w*xx)
    wxsq_sum = np.sum(w*(xx**2))

    b_denom = w_sum-wx_sum*wx_sum/wxsq_sum
    b_num = wy_sum-wx_sum*wxy_sum/wxsq_sum
    if np.abs(b_denom)<1.0e-20:
        assert np.abs(b_num)<1.0e-10
        return 0.0, 0.0
    else:
        b = b_num/b_denom

    m_denom = wxsq_sum
    m_num = (wxy_sum-b*wx_sum)
    m = m_num/m_denom

    return m, b

def cdf_to_pdf(cdf_x, cdf_y):

    pdf_x = np.copy(cdf_x)
    pdf_y = np.zeros(pdf_x.shape, dtype=float)
    dx = (cdf_x.max()-cdf_x.min())/len(cdf_x)
    for ix in range(len(pdf_x)):
        x0 = pdf_x[ix]
        m, b = _fit_with_wgt(pdf_x[ix],
                             cdf_x,
                             cdf_y,
                             dx)
        pdf_y[ix] = m
    delta = np.diff(pdf_x).min()
    if delta <= 0.0:
        raise RuntimeError(f'delta = {delta} in cdf_to_pdf')
    xmin = pdf_x[0]-0.5*delta
    xmax = pdf_x[-1]+0.5*delta
    pdf_x = np.concatenate([np.array([xmin]),
                            pdf_x,
                            np.array([xmax])])
    if not (np.diff(pdf_x)>0).all():
        raise RuntimeError(f'pdf_x diff min {np.diff(pdf_x).min()} '
                           'in cdf to pdf')
    pdf_y = np.concatenate([np.array([0.0]),
                            pdf_y,
                            np.array([0.0])])

    integral = 0.5*(pdf_x[1:]-pdf_x[:-1])*(pdf_y[1:]+pdf_y[:-1])
    integral = integral.sum()

    return pdf_x, pdf_y/integral


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
