import matplotlib

matplotlib.use("agg")
import logging
from pathlib import Path
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import solve_banded


def debug_plot(
    file_name: Union[str, Path],
    roi_trace: np.ndarray,
    neuropil_trace: np.ndarray,
    corrected_trace: np.ndarray,
    r: float,
    r_vals: np.ndarray = None,
    err_vals: np.ndarray = None,
) -> None:
    """Create debug plot for neuropil correction"""
    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(211)
    ax.plot(roi_trace, "r", label="raw")
    ax.plot(corrected_trace, "b", label="fc")
    ax.plot(neuropil_trace, "g", label="neuropil")
    ax.set_xlim(0, roi_trace.size)
    ax.set_title(
        "raw(%.02f, %.02f) fc(%.02f, %.02f) r(%f)"
        % (
            roi_trace.min(),
            roi_trace.max(),
            corrected_trace.min(),
            corrected_trace.max(),
            r,
        )
    )
    ax.legend()

    if r_vals is not None:
        ax = fig.add_subplot(212)
        ax.plot(r_vals, err_vals, "o")

    plt.savefig(file_name)
    plt.close()


def get_diagonals_from_sparse(mat: sparse) -> dict:
    """Returns a dictionary of diagonals keyed by offsets

    Parameters
    ----------
    mat: scipy.sparse
        matrix

    Returns
    -------
    dict:
        diagonals keyed by offsets
    """

    mat_dia = mat.todia()  # make sure the matrix is in diagonal format

    offsets = mat_dia.offsets
    diagonals = mat_dia.data

    mat_dict = {}

    for i, o in enumerate(offsets):
        mat_dict[o] = diagonals[i]

    return mat_dict


def ab_from_diagonals(mat_dict: dict) -> np.ndarray:
    """Constructs value for scipy.linalg.solve_banded

    Parameters
    ----------
    mat_dict: dict
        dictionary of diagonals keyed by offsets

    Returns
    -------
    ab: np.ndarray
        value for scipy.linalg.solve_banded
    """
    offsets = list(mat_dict.keys())
    l = -np.min(offsets)
    u = np.max(offsets)

    T = mat_dict[offsets[0]].shape[0]

    ab = np.zeros([l + u + 1, T])

    for o in offsets:
        index = u - o
        ab[index] = mat_dict[o]

    return ab


def error_calc(
    F_M: np.ndarray,
    F_N: np.ndarray,
    F_C: np.ndarray,
    r: float,
) -> float:
    """Calculate root mean square error between corrected trace and roi trace with
    subtracted neuropil contamination.

    Parameters
    -----------
    F_M: np.ndarray
        ROI trace
    F_N: np.ndarray
        neuropil trace
    F_C: np.ndarray
        neuropil corrected trace
    r: float
        contamination ratio

    Returns
    --------
    er: float
        RMSE
    """
    er = np.sqrt(np.mean(np.square(F_C - (F_M - r * F_N)))) / np.mean(F_M)

    return er


def ab_from_T(T: int, lam: float, dt: float):
    # using csr because multiplication is fast
    Ls = -sparse.eye(T - 1, T, format="csr") + sparse.eye(
        T - 1, T, 1, format="csr"
    )
    Ls /= dt
    Ls2 = Ls.T.dot(Ls)

    M = sparse.eye(T) + lam * Ls2
    mat_dict = get_diagonals_from_sparse(M)
    ab = ab_from_diagonals(mat_dict)

    return ab


class NeuropilSubtract(object):
    """Class to jointly estimate the corrected fluorescence
    trace (F_C) and the R contamination value given
    a neuropil trace and ROI trace

    The estimation is performed with a gradient descent to
    minimize a loss function:

    E = (F_C - (F_M - r*F_n))**2 + lam*dt*F_C

    Parameters
    ----------
    lam: float
        lambda weight
    dt: float
    folds int
    """

    def __init__(self, lam: float = 0.05, dt: float = 1.0, folds: int = 4):
        self.lam = lam
        self.dt = dt
        self.folds = folds

        self.T = None
        self.T_f = None
        self.ab = None

        self.F_M = None
        self.F_N = None

        self.r_vals = None
        self.error_vals = None
        self.r = None
        self.error = None

    def set_F(self, F_M: np.ndarray, F_N: np.ndarray) -> None:
        """Break the F_M and F_N traces into the number of folds specified
        in the class constructor and normalize each fold of F_M and R_N relative to F_N.
        """

        F_M_len = len(F_M)
        F_N_len = len(F_N)

        if F_M_len != F_N_len:
            raise Exception(
                "F_M and F_N must have the same length (%d vs %d)"
                % (F_M_len, F_N_len)
            )

        if self.T != F_M_len:
            logging.debug("updating ab matrix for new T=%d", F_M_len)
            self.T = F_M_len
            self.T_f = int(self.T / self.folds)
            self.ab = ab_from_T(self.T_f, self.lam, self.dt)

        self.F_M = []
        self.F_N = []

        for fi in range(self.folds):
            self.F_M.append(F_M[fi * self.T_f : (fi + 1) * self.T_f])
            self.F_N.append(F_N[fi * self.T_f : (fi + 1) * self.T_f])

    def fit_block_coordinate_desc(
        self, r_init: float = 5.0, min_delta_r: float = 0.00000001
    ) -> None:
        F_M = np.concatenate(self.F_M)
        F_N = np.concatenate(self.F_N)

        r_vals = []
        error_vals = []
        r = r_init

        delta_r = None
        it = 0

        ab = ab_from_T(self.T, self.lam, self.dt)
        while delta_r is None or delta_r > min_delta_r:
            F_C = solve_banded((1, 1), ab, F_M - r * F_N)
            new_r = -np.sum((F_C - F_M) * F_N) / np.sum(np.square(F_N))
            error = self.estimate_error(new_r)

            error_vals.append(error)
            r_vals.append(new_r)

            if r is not None:
                delta_r = np.abs(r - new_r) / r

            r = new_r
            it += 1

        self.r_vals = r_vals
        self.error_vals = error_vals
        self.r = r_vals[-1]
        self.error = error_vals.min()

    def fit(
        self,
        r_range: List[float, float] = [0.0, 2.0],
        iterations: int = 3,
        dr: float = 0.1,
        dr_factor: float = 0.1,
    ) -> None:
        """Estimate error values for a range of r values.  Identify a new r range
        around the minimum error values and repeat multiple times.
        TODO: docs
        """
        global_min_error = None
        global_min_r = None

        r_vals = []
        error_vals = []

        it_range = r_range
        it = 0

        it_dr = dr
        while it < iterations:
            it_errors = []

            # build a set of r values evenly distributed in a current range
            rs = np.arange(it_range[0], it_range[1], it_dr)

            # estimate error for each r
            for r in rs:
                error = self.estimate_error(r)
                it_errors.append(error)

                r_vals.append(r)
                error_vals.append(error)

            # find the minimum in this range and update the global minimum
            min_i = np.argmin(it_errors)
            min_error = it_errors[min_i]

            if global_min_error is None or min_error < global_min_error:
                global_min_error = min_error
                global_min_r = rs[min_i]

            logging.debug(
                "iteration %d, r=%0.4f, e=%.6e",
                it,
                global_min_r,
                global_min_error,
            )

            # if the minimum error is on the upper boundary,
            # extend the boundary and redo this iteration
            if min_i == len(it_errors) - 1:
                logging.debug(
                    "minimum error found on upper r bound, extending range"
                )
                it_range = [rs[-1], rs[-1] + (rs[-1] - rs[0])]
            else:
                # error is somewhere on either side of the minimum error index
                it_range = [
                    rs[max(min_i - 1, 0)],
                    rs[min(min_i + 1, len(rs) - 1)],
                ]
                it_dr *= dr_factor
                it += 1

        self.r_vals = r_vals
        self.error_vals = error_vals
        self.r = global_min_r
        self.error = global_min_error

    def estimate_error(self, r: float) -> float:
        """Estimate error values for a given r for each fold and return the mean."""

        errors = np.zeros(self.folds)
        for fi in range(self.folds):
            F_M = self.F_M[fi]
            F_N = self.F_N[fi]
            F_C = solve_banded((1, 1), self.ab, F_M - r * F_N)
            errors[fi] = abs(error_calc(F_M, F_N, F_C, r))

        return np.mean(errors)


def estimate_contamination_ratios(
    F_M: np.ndarray,
    F_N: np.ndarray,
    lam: float = 0.05,
    folds: int = 4,
    iterations: int = 3,
    r_range: List[float, float] = [0.0, 2.0],
    dr: float = 0.1,
    dr_factor: float = 0.1,
):
    """Calculates neuropil contamination of ROI

    Parameters
    ----------
       F_M: ROI trace
       F_N: Neuropil trace

    Returns
    -------
    dictionary: key-value pairs
        * 'r': the contamination ratio -- corrected trace = M - r*N
        * 'err': RMS error
        * 'min_error': minimum error
        * 'bounds_error': boolean. True if error or R are outside tolerance
    """

    ns = NeuropilSubtract(lam=lam, folds=folds)

    ns.set_F(F_M, F_N)

    ns.fit(r_range=r_range, iterations=iterations, dr=dr, dr_factor=dr_factor)

    # ns.fit_block_coordinate_desc()

    if ns.r < 0:
        logging.warning("r is negative (%f). return 0.0.", ns.r)
        ns.r = 0

    return {
        "r": ns.r,
        "r_vals": ns.r_vals,
        "err": ns.error,
        "err_vals": ns.error_vals,
        "min_error": ns.error,
        "it": len(ns.r_vals),
    }
