import multiprocessing
import numpy as np
from typing import Tuple
from functools import partial
from FastLZeroSpikeInference import fast

from ophys_etl.modules.event_detection.utils import count_and_minmag


def fast_lzero(penalty: float, dat: np.ndarray, gamma: float,
               constraint: bool) -> np.ndarray:
    """runs fast spike inference and returns an array like the input trace
    with data substituted by event magnitudes

    Parameters
    ----------
    penalty: float
        tuning parameter lambda > 0
    dat: np.ndarray
        fluorescence data
    gamma: float
        a scalar value for the AR(1) decay parameter; 0 < gamma <= 1
    constraint: bool
        constrained (true) or unconstrained (false) optimization

    Returns
    -------
    out: np.ndarray
        event magnitude array, same shape as dat

    Notes
    -----
    https://github.com/jewellsean/FastLZeroSpikeInference/blob/cdfaade68ceb6aa15ec5003c460de4e0575f1d5f/python/FastLZeroSpikeInference/fast.py#L30  # noqa: E501

    """
    ev = fast.estimate_spikes(dat, gamma, penalty,
                              constraint, estimate_calcium=True)
    out = np.zeros(ev['dat'].shape)
    out[ev['spikes']] = ev['pos_spike_mag']
    return out


def fast_lzero_regularization_search_bracket(
        trace: np.ndarray, min_size: float, gamma: float,
        base_penalty: float = 0, penalty_step: float = 0.1,
        step_min: float = 0.0001,
        bisect=False) -> Tuple[np.ndarray, np.ndarray]:
    """finds events through a regularization search, subject to the
    constraints of having n_events > 0 (establishing the upper bound for
    regularization penalty) and the smallest event magnitude near the noise
    estimate.

    Parameters
    ----------
    trace: np.ndarray
        the 1D trace data
    min_size: float
        a target for the minimum amplitude magnitude. This is typically a
        multiple of an estimate of the noise.
    gamma: float
        a scalar value for the AR(1) decay parameter; 0 < gam <= 1
    base_penalty: float
        initial guess for the regularization penalty
    penalty_step: float
        penalty step size
    step_min: float
        minimum possible step size
    bisect: bool
        when True, invokes a large step

    Returns
    -------
    events: np.ndarray
        array of event magnitudes, the same size as trace
    penalty: float
        the optimized regularization parameter

    Notes
    -----
    - This is the original Allen 'bracket' routine refactored to make it
    readable. The logic has not been changed.
    - In original, a multiplicative factor for the noise estimate was
    parametrized, but never used. Hard-coded here, for now, as the original
    default of 1.0.
    - In original, L0_contstrain (bool) was parametrized to tell FastLZero to
    perform LZero regularization, which was always True. Hard-coded
    here to True.

    """
    L0_constrain = True

    # these args are immutable through the recursion
    recursive_partial = partial(fast_lzero_regularization_search_bracket,
                                trace, min_size, gamma,
                                step_min=step_min)

    # evaluate fast_lzero at one point
    penalty = base_penalty + penalty_step
    if penalty < penalty_step:
        penalty = penalty_step
        base_penalty += penalty_step
    flz_eval = fast_lzero(penalty, trace, gamma, L0_constrain)
    n_events, min_event_mag = count_and_minmag(flz_eval)

    if n_events == 0 and bisect is True:
        base_penalty -= 5 * penalty_step
        return recursive_partial(
                base_penalty=base_penalty,
                penalty_step=penalty_step)

    if penalty_step == step_min:
        if min_event_mag > min_size and bisect is True:
            base_penalty -= 5 * penalty_step
            return recursive_partial(
                    base_penalty=base_penalty,
                    penalty_step=penalty_step)
        else:
            while n_events > 0 and min_event_mag < min_size:
                last_flz_eval = np.copy(flz_eval)
                penalty += penalty_step
                flz_eval = fast_lzero(penalty, trace, gamma, L0_constrain)
                n_events, min_event_mag = count_and_minmag(flz_eval)
            if n_events == 0:
                return (last_flz_eval, penalty - penalty_step)
            else:
                return (flz_eval, penalty)

    if n_events == 0 and bisect is False:
        base_penalty += 0.5 * penalty_step - penalty_step / 10
        penalty_step /= 10
        return recursive_partial(
                base_penalty=base_penalty,
                penalty_step=penalty_step,
                bisect=True)

    if n_events > 0 and min_event_mag < min_size:
        return recursive_partial(
                base_penalty=penalty,
                penalty_step=penalty_step)

    if n_events > 0 and min_event_mag > min_size and penalty_step > step_min:
        if bisect:
            base_penalty -= 5 * penalty_step
        else:
            base_penalty += 0.5 * penalty_step - penalty_step / 10
            penalty_step /= 10
        return recursive_partial(
                base_penalty=base_penalty,
                penalty_step=penalty_step,
                bisect=not bisect)


def get_events(traces: np.ndarray, noise_estimates: np.ndarray,
               gamma: float, ncpu: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """finds events through one of two methods for regularization search.
    parallelized by trace.

    Parameters
    ----------
    traces: np.ndarray
        ntrace x nframes, the trace data
    noise_estimates: np.ndarray
        length ntrace, estimates of standard deviations
    gamma: float
        decay-time dependent parameter, passed to FastLZero algorithm. The
        attenuation factor from exponential decay in one sampling step.
    ncpu: int
        number or workers for the multiprocessing pool

    Returns
    -------
    events: np.ndarray
        event magnitudes. same shape as traces. zero where no event detected.
    lambdas: np.ndarray
        length ntrace optimized regularization values.

    """
    args = [(trace, sigma, gamma)
            for trace, sigma in zip(traces, noise_estimates)]
    func = fast_lzero_regularization_search_bracket

    # pytest-cov does not play nice with Pool context manager so explicit
    # close/join to get code coverage report in subprocess calls
    pool = multiprocessing.Pool(ncpu)
    results = pool.starmap(func, args)
    pool.close()
    pool.join()

    events, lambdas = zip(*results)
    events = np.array(events)
    lambdas = np.array(lambdas)
    return events, lambdas
