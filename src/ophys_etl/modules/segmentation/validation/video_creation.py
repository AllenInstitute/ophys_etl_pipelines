import numpy as np
from typing import Tuple, Optional, Union
from skimage.draw import ellipse
from scipy.optimize import bisect
from scipy.stats import pearsonr

from ophys_etl.types import ExtractROI
from ophys_etl.modules.event_detection import validation
from ophys_etl.modules.event_detection.utils import trace_noise_estimate


def create_roi_ellipse(center: Tuple[int, int],
                       r_radius: float,
                       c_radius: float,
                       rotation: float,
                       shape: Tuple[int, int],
                       id: int = 0) -> ExtractROI:
    """create an elliptical ROI

    Parameters
    ----------
    center: Tuple[int, int]
        the global FOV (row, col) coordinates of the center pixel
    r_radius: float
        the radius of the ellipse across rows
    c_radius: float
        the radius of the ellipse across columns
    rotation: float
        the rotation of the radii axes [-pi, pi]
    shape: Tuple[int, int]
        the FOV shape
    id: int
        the id for the resulting ROI

    Returns
    -------
    roi: ExtractROI
        the ROI

    """

    rows, cols = ellipse(r=center[0],
                         c=center[1],
                         r_radius=r_radius,
                         c_radius=c_radius,
                         rotation=rotation,
                         shape=shape)

    r0 = rows.min()
    c0 = cols.min()
    height = rows.ptp() + 1
    width = cols.ptp() + 1

    mask = np.zeros((height, width), dtype=bool)
    for r, c in zip(rows, cols):
        mask[r - r0, c - c0] = True
    mask = [i.tolist() for i in mask]

    roi = ExtractROI(id=id,
                     x=c0,
                     y=r0,
                     width=width,
                     height=height,
                     valid=True,
                     mask=mask)
    return roi


def polynomial_weight_mask(
        roi: ExtractROI,
        order: int,
        seed: Optional[Union[int, np.random.Generator]] = None
        ) -> np.ndarray:
    """for a given ROI, calculate intensities, with maximum value 1.0
    that have polynomial variations up to an including the specified
    order.

    Parameters
    ----------
    roi: ExtractROI
        an roi instance
    order: int
        the polynomial order for the weight surface
    seed: int or None or np.random.Generator
        passed as seed to np.random.default_rng

    Returns
    -------
    weight: np.ndarray

    """
    coeffs = []
    order_indices = np.arange(order + 1)

    rng = np.random.default_rng(seed=seed)

    for order_index in order_indices:
        if order_index == 0:
            coeffs.append([1.0])
        else:
            discount = np.power(10.0, -(order_index + 0.5))
            coeffs.append(rng.standard_normal(order_index + 1) * discount)
    cmat = np.zeros((order + 1, order + 1))
    for c in coeffs:
        ir = len(c) - 1
        ic = 0
        for citem in c:
            cmat[ir, ic] = citem
            ir -= 1
            ic += 1

    coords = np.argwhere(roi["mask"])
    center = coords.mean(axis=0).astype(int)
    centered_coords = coords - center
    weights = np.polynomial.polynomial.polyval2d(centered_coords[:, 0],
                                                 centered_coords[:, 1],
                                                 cmat)
    weight = np.zeros_like(roi["mask"]).astype(float)
    for (r, c), w in zip(coords, weights):
        weight[r, c] = w
    weight = weight / weight.max()
    return weight


def correlated_trace(common_trace: np.ndarray,
                     correlation_target: float,
                     seed: Optional[Union[int, np.random.Generator]] = None
                     ) -> np.ndarray:
    """adds noise to a trace to acheive the specified correlation

    Parameters
    ----------
    common_trace: np.ndarray
        a 1D array against which traces with correlations specified
        by weights will be generated.
    correlation_target: float
        a Pearson correlation coefficient in (0.0, 1.0)
    seed: int or None or np.random.Generator
        passed as seed to np.random.default_rng


    Returns
    -------
    new_trace: np.ndarray
        the correlated trace

    """
    if (correlation_target <= 0.0) | (correlation_target >= 1.0):
        raise ValueError("correlation must be in range (0.0, 1.0) "
                         f"{correlation_target} was provided.")

    rng = np.random.default_rng(seed=seed)
    noise_base = rng.standard_normal(common_trace.size)

    def noisy_trace(trace, factor):
        new_trace = trace + noise_base * factor
        return new_trace.astype(trace.dtype)

    def correlation_callable(factor):
        if factor == 0.0:
            pearson = 1.0
        else:
            candidate = noisy_trace(common_trace, factor)
            pearson = pearsonr(common_trace, candidate)[0]
        return pearson - correlation_target

    # solve by bisection
    lower_factor_limit = 1.0
    upper_factor_limit = 1.0
    lower_val = 1.0
    upper_val = 1.0
    while np.sign(lower_val) == np.sign(upper_val):
        lower_factor_limit /= 10.0
        upper_factor_limit *= 10.0
        lower_val = correlation_callable(lower_factor_limit)
        upper_val = correlation_callable(upper_factor_limit)

    result = bisect(f=correlation_callable,
                    a=lower_factor_limit,
                    b=upper_factor_limit)
    new_trace = noisy_trace(common_trace, result)
    return new_trace


def correlated_traces_from_weights(
        common_trace: np.ndarray,
        weights: np.ndarray,
        seed: Optional[Union[int, np.random.Generator]] = None
        ) -> np.ndarray:
    """generates correlated traces with correlations given by weights

    Parameters
    ----------
    common_trace: np.ndarray
        a 1D array against which traces with correlations specified
        by weights will be generated.
    weights: np.ndarray
        (nrows x ncols) array where the intensities represent the
        desire Pearson correlation coefficient relative to an
        imaginary common-mode trace.
    seed: int or None or np.random.Generator
        passed as seed to np.random.default_rng

    Returns
    -------
    traces:  np.ndarray
        (nframes x nrows x ncols), uint16 traces

    Notes
    -----
    for any (i, j) where weights[i, j] is zero, the corresponding
    traces[:, i, j] will be all zeros.

    """
    traces = np.empty(shape=(common_trace.shape[0], *weights.shape),
                      dtype=common_trace.dtype)
    rng = np.random.default_rng(seed=seed)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            if weights[i, j] != 0.0:
                traces[:, i, j] = correlated_trace(common_trace,
                                                   weights[i, j],
                                                   seed=rng)
            else:
                traces[:, i, j] = 0.0
    return traces


def movie_with_fake_rois(spacing: int,
                         shape: Tuple[int, int, int],
                         correlation_low: float,
                         correlation_high: float,
                         r_radius: float,
                         c_radius: float,
                         rotation: float,
                         n_events: int,
                         rate: float = 11.0,
                         decay_time: float = 0.4,
                         seed: Optional[Union[int, np.random.Generator]] = None
                         ) -> np.ndarray:
    """
    create a movie (3D numpy array) with faked signals and correlations.
    ROI copies will be created on a grid, with a progressively changing amount
    of correlation (i.e. chaning signal-to-noise ratio).

    Parameters
    ----------
    spacing: int
        the row and column spacing between ROI centers (in pixels)
    shape: Tuple[int, int, int]
        the nframes x nrows x ncols shape of the desired output movie
    correlation_low: float
        the correlation for the lowest SNR ROI
    correlation_high: float
        the correlation for the highest SNR ROI
    r_radius: float
        the radius of the ROIs across rows, before rotation
    c_radius: float
        the radius of the ROIs across columns, before rotation
    rotation: float
        the rotation of the radii axes [-pi, pi]
    n_events: the number of events to simulate for the common-mode
        trace of each ROI.
    rate: float
        the sampling rate of the data [Hz], i.e. 11.0 for mesoscope-like
    decay_time: float
        the fluorescence decay time [seconds]
    seed: int or None or np.random.Generator
        passed as seed to np.random.default_rng

    Returns
    -------
    traces: np.ndarray
        with shape determined by parameter 'shape'

    """
    rng = np.random.default_rng(seed=seed)

    # create a grid of ROIs of the same shape
    nrow = int(np.floor(shape[1] / spacing))
    ncol = int(np.floor(shape[2] / spacing))
    rois = []
    roi_id = 1
    roi_summary = dict()
    for ir in range(nrow):
        for ic in range(ncol):
            r = int(spacing / 2 + ir * spacing)
            c = int(spacing / 2 + ic * spacing)
            rois.append(create_roi_ellipse(center=(r, c),
                                           r_radius=r_radius,
                                           c_radius=c_radius,
                                           rotation=rotation,
                                           shape=shape[1:],
                                           id=roi_id))
            roi_summary[roi_id] = {
                    "center": (r, c)}
            roi_id += 1

    # determine a weight factor for each ROI
    weights = polynomial_weight_mask(rois[0], 2, seed=rng)
    wfactors = np.linspace(correlation_high,
                           correlation_low,
                           len(rois))

    # simulate traces for each pixel in each ROI
    traces = np.zeros(shape=shape, dtype="uint16")
    t = np.arange(shape[0])
    for wfactor, roi in zip(wfactors, rois):
        rng.shuffle(t)
        magnitudes = rng.standard_normal(size=n_events)
        # for this ROI, simulate a common-mode trace
        trace = validation.sum_events(n_samples=shape[0],
                                      timestamps=t[0:n_events],
                                      magnitudes=magnitudes,
                                      decay_time=decay_time,
                                      rate=rate)

        # for every pixel in this ROI, create a trace with a specified
        # correlation to the common-mode
        roi_traces = correlated_traces_from_weights(
                common_trace=trace,
                weights=(weights * wfactor),
                seed=rng)

        # for any non-zero trace in this ROI, estimate the noise
        # and get the minimum value
        not_empty_indices = np.argwhere(weights != 0)
        noises = []
        tmins = []
        for indices in not_empty_indices:
            trace = roi_traces[:, indices[0], indices[1]]
            noises.append(
                    trace_noise_estimate(trace, filt_length=33))
            tmins.append(trace.min())
        noise = np.mean(noises)
        tmin = np.mean(tmins)

        # convert to uint16 with max value 2**12 and offset 500
        tmin = roi_traces.min()
        tptp = roi_traces.ptp()
        toff = 500
        roi_traces = ((roi_traces - tmin) * np.power(2, 12) / tptp) + toff
        roi_traces = roi_traces.astype("uint16")
        roi_summary[roi["id"]].update(
                {
                    "noise": noise,
                    "tmin": tmin,
                    "tptp": tptp,
                    "toff": toff})

        # place the traces for this ROI in the global movie
        for indices in not_empty_indices:
            traces[:, roi["y"] + indices[0], roi["x"] + indices[1]] = \
                    roi_traces[:, indices[0], indices[1]]

    # for all the ROIs, determine the average offsets, noise etc
    mins = []
    ptps = []
    offs = []
    noises = []
    for roi_s in roi_summary.values():
        mins.append(roi_s["tmin"])
        ptps.append(roi_s["tptp"])
        offs.append(roi_s["toff"])
        noises.append(roi_s["noise"])
    tmin = np.mean(mins)
    tptp = np.mean(ptps)
    toff = np.mean(offs)
    noise = np.mean(noises)

    # fill in the non-ROI pixels with some noise, to be a little realistic
    empty_indices = np.argwhere(traces.ptp(axis=0) == 0.0)
    for r, c in empty_indices:
        tmp = np.random.randn(shape[0]) * noise
        traces[:, r, c] = ((tmp - tmin) * np.power(2, 12) / tptp) + toff

    return traces
