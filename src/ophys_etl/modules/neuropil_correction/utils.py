import logging
from pathlib import Path
from typing import List, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
from scipy.linalg import solve_banded

matplotlib.use("agg")


def debug_plot(
    file_name: Union[str, Path],
    roi_trace: np.ndarray,
    neuropil_trace: np.ndarray,
    corrected_trace: np.ndarray,
    r: float,
    r_vals: np.ndarray = None,
    err_vals: np.ndarray = None,
) -> None:
    """Create debug plot for neuropil correction

    Parameters
    -----------
    file_name: Union[str, Path]
        path to output plot image
    roi_trace: np.ndarray
        roi trace for a single roi
    neuropil_trace: np.ndarray
        neuropil trace for a single roi
    corrected_trace: np.ndarray
        corrected trace for a single roi
    r: float
        r contamination value
    r_vals: np.ndarray
        range of r values fitted
    err_vals: np.ndarray
        error values associated with range of r values
    """
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


def fill_unconverged_r(
    corrected_neuropil_traces: np.ndarray,
    roi_traces: np.ndarray,
    neuropil_traces: np.ndarray,
    r_array: np.ndarray,
    flag_threshold: float = 1.0,
    fill_r_val: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """R values that are unconverged as defined by values that exceed 1.0
    are filled in with the mean value of all other cells from the same
    experiment. The corrected fluorescence trace is recalculated with the
    adjusted R value.

    Parameters
    ----------
    corrected_neuropil_traces: np.ndarray
        corrected traces for all ROIs in an experiment
    roi_traces: np.ndarray
        ROI traces for all ROIs in an experiment
    neuropil_traces: np.ndarray
        neuropil traces for all ROIs in an experiment
    r_array: np.ndarray
        array of r values for all ROIs in an experiment
    flag_threshold: float = 1.0
        threshold r value to flag for filling
    fill_r_val: float = 0.7
        default fill value if less than 5 ROIs with r<1

    Returns
    -------

    """
    flagged_mask = r_array > flag_threshold
    if r_array[(~flagged_mask) & (r_array > 0)].shape[0] >= 5:
        fill_r_val = r_array[(~flagged_mask) & (r_array > 0)].mean()
    r_array[flagged_mask] = fill_r_val
    corrected_neuropil_traces[flagged_mask] = (
        roi_traces[flagged_mask] - fill_r_val * neuropil_traces[flagged_mask]
    )
    rmse = []
    for F_M, F_N, r in zip(roi_traces, neuropil_traces, r_array):
        ns = NeuropilSubtract()
        ns.set_F(F_M, F_N)
        rmse.append(ns.estimate_error(r))
    return corrected_neuropil_traces, r_array, np.array(rmse)


class NeuropilSubtract(object):
    """Class to jointly estimate the corrected fluorescence
    trace (F_C) and the R contamination value given
    a neuropil trace and ROI trace

    The estimation is performed with a gradient descent to
    minimize a loss function:

    E = (F_C - (F_M - r*F_n))**2 + lam*dt*F_C

    Parameters
    ----------
    F_M_array: np.ndarray
        ROI trace
    F_N: np.ndarray
        neuropil trace
    F_C: np.ndarray
        neuropil corrected trace
    r: float
        contamination ratio
    lam: float
        lambda weight
    dt: float
    folds: int
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

    def __get_diagonals_from_sparse(self, mat: sparse) -> dict:
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

    def ab_from_diagonals(self, mat_dict: dict) -> np.ndarray:
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
        ll = -np.min(offsets)
        u = np.max(offsets)

        T = mat_dict[offsets[0]].shape[0]

        ab = np.zeros([ll + u + 1, T])

        for o in offsets:
            index = u - o
            ab[index] = mat_dict[o]

        return ab

    def __error_calc(
        self,
        F_M: np.ndarray,
        F_N: np.ndarray,
        F_C: np.ndarray,
        r: float,
    ) -> float:
        """Calculate root mean square error between corrected trace
        and roi trace with subtracted neuropil contamination.

        Parameters
        -----------
        F_M_array: np.ndarray
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

    def __ab_from_T(self, T: int, lam: float, dt: float) -> np.ndarray:
        """

        Parameters
        ----------
        T: int
        lam: float
        dt: float

        Returns
        -------
        np.ndarray
        """
        # using csr because multiplication is fast
        Ls = -sparse.eye(T - 1, T, format="csr") + sparse.eye(
            T - 1, T, 1, format="csr"
        )
        Ls /= dt
        Ls2 = Ls.T.dot(Ls)

        M = sparse.eye(T) + lam * Ls2
        mat_dict = self.__get_diagonals_from_sparse(M)
        ab = self.ab_from_diagonals(mat_dict)

        return ab

    def set_F(self, F_M: np.ndarray, F_N: np.ndarray) -> None:
        """Break the F_M and F_N traces into the number of folds specified
        in the class constructor and normalize each fold of F_M and R_N
        relative to F_N.

        Parameters:
        -----------
        F_M: np.ndarray
            ROI trace
        F_N: np.ndarray
            neuropil trace
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
            self.ab = self.__ab_from_T(self.T_f, self.lam, self.dt)

        self.F_M = []
        self.F_N = []

        for fi in range(self.folds):
            self.F_M.append(F_M[fi * self.T_f : (fi + 1) * self.T_f])  # noqa
            self.F_N.append(F_N[fi * self.T_f : (fi + 1) * self.T_f])  # noqa

    def fit(
        self,
        r_range: List[float] = [0.0, 2.0],
        iterations: int = 3,
        dr: float = 0.1,
        dr_factor: float = 0.1,
    ) -> None:
        """Estimate error values for a range of r values.
        Identify a new r range around the minimum error
        values and repeat multiple times.

        Parameters:
        -----------
        r_range: List[float]
            range of r values to search for minimized error
        iterations: int = 3
            number of iterations to search with lowered
            step size
        dr: float = 0.1
            initial step size
        dr_factor: float = 0.1
            step size factor for each iteration

        Returns
        --------
        None

        Function sets instance variables
        self.r_vals: list
            r values searched in fitting
        self.error_vals: list
            errors for each r value searched
        self.r: float
            r value at the global minimum within r_range
        self.error: float
            error at global minimum r
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
        """Estimate error values for a given r for each fold
        and return the mean.

        Parameters
        ----------
        r: float

        Returns
        --------
        error: float
        """

        errors = np.zeros(self.folds)
        for fi in range(self.folds):
            F_M = self.F_M[fi]
            F_N = self.F_N[fi]
            F_C = solve_banded((1, 1), self.ab, F_M - r * F_N)
            errors[fi] = abs(self.__error_calc(F_M, F_N, F_C, r))

        return np.mean(errors)


def estimate_contamination_ratios(
    F_M: np.ndarray,
    F_N: np.ndarray,
    lam: float = 0.05,
    folds: int = 4,
    iterations: int = 3,
    r_range: List[float] = [0.0, 2.0],
    dr: float = 0.1,
    dr_factor: float = 0.1,
):
    """Calculates neuropil contamination of ROI

    Parameters
    ----------
    F_M: np.ndarray
        roi trace
    F_N: np.ndarray
        neuropil trace
    lam: float = 0.05
        weight of smoothness constraint of loss function
    folds: int = 4
        number of folds to split data
    iterations: int = 3
        number of iterations to search with lowered
        step size
    r_range: List[float]
        range of r values to search for minimized error
    dr: float = 0.1
        initial step size
    dr_factor: float = 0.1
        step size factor for each iteration

    Returns
    -------
    dict: key-value pairs
        r: the contamination ratio -- corrected trace = M - r*N
        r_vals: range of r values fitted
        err_vals: error values associated with range of r values
        err: error at r
        min_error: minimum error = error at r
        it: number of iterations


    """

    ns = NeuropilSubtract(lam=lam, folds=folds)

    ns.set_F(F_M, F_N)

    ns.fit(r_range=r_range, iterations=iterations, dr=dr, dr_factor=dr_factor)

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
