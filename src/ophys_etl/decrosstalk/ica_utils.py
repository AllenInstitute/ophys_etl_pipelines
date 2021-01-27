from typing import List, Tuple, Union

import numpy as np
import copy
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA

__all__ = ["whiten_data",
           "fix_source_assignment",
           "pearson_ica_in_to_out",
           "run_ica"]


def pearson_ica_in_to_out(signal_in: np.ndarray,
                          signal_out: np.ndarray) -> List[float]:
    """
    Function to compute correlations between two vectors (traces in our case)

    Parameters:
    -----------

    signal_in -- a 2xN numpy array representing the input signals.
    N is the number of timesteps. signal_in[0,:] is the input
    'signal'; signal_in[1,:] is the input 'crosstalk'

    signal_out -- a 2xN numpy array representing the output signals.
    N is the number of timesteps. signal_in[0,:] is the output
    'signal'; signal_in[1,:] is the output 'crosstalk'

    Returns
    -------
    A list of Pearson's R correlation coefficents;
        [Corr(signal_in[0,:], signal_out[0,:]),
         Corr(signal_in[1,:], signal_out[0,:]),
         Corr(signal_in[0,:], signal_out[1,:]),
         Corr(signal_in[1,:], signal_out[1,:])]
    """

    if signal_in.shape != signal_out.shape:
        msg = "\nShape of inputs doesn't align\n"
        msg += "singal_in %s\n" % str(signal_in.shape)
        msg += "signal_out %s\n" % str(signal_out.shape)
        raise RuntimeError(msg)

    if signal_in.shape[0] >= signal_in.shape[1]:
        msg = "\nShape of input is reversed: "
        msg += "%s\nuse input.T" % str(signal_in.shape)
        raise RuntimeError(msg)

    cor_0_0, _ = pearsonr(signal_out[0, :], signal_in[0, :])
    cor_0_1, _ = pearsonr(signal_out[0, :], signal_in[1, :])
    cor_1_0, _ = pearsonr(signal_out[1, :], signal_in[0, :])
    cor_1_1, _ = pearsonr(signal_out[1, :], signal_in[1, :])
    return [cor_0_0, cor_0_1, cor_1_1, cor_1_0]


def whiten_data(x: np.ndarray) -> Tuple[np.ndarray,
                                        np.ndarray,
                                        np.ndarray]:
    """
    Function to debias (subtract mean) and whiten* the data

    :param x:  -- shaped NxM where N is the number of timesteps
                  per signal and M is the number of signals

    :return:
        whitened_data  -- shaped NxM

        whitening matrix -- shaped MxM

        the mean of the columns of data -- shaped M

    whitened data is such that np.corrcoef(whitened_data.transpose())
    is approximately the MxM identity matrix

    whitening_matrix is the transformation such that
    np.dot(x, x-mean) = whitened_data
    """
    m = np.mean(x, axis=0)
    x = x - m
    n = np.sqrt(x.shape[0])
    u, s, v = np.linalg.svd(x, full_matrices=False)
    w = np.dot(v.T / s, v) * n
    return np.dot(u, v) * n, w, m


def fix_source_assignment(ica_input: np.ndarray,
                          ica_output: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Function to rearrange the rows of ica_output so that
    the first row is the one that most strongly resembles
    the the first row of ica_input

    Parameters
    ----------
    ica_input -- an NxM numpy array where N is the number of
    channels (2; one for "signal", one for "crosstalk") and
    M is the number of timesteps. This was the input to the
    ICA decomposition.

    ica_output -- an NxM numpy array. This was the output of
    the initial ICA decomposition

    Returns
    -------
    An NxM numpy array created by rearranging (if necessary)
    the rows of ica_output.

    A boolean that is true if the rows of ica_output had to
    be swapped.
    """
    corrs = pearson_ica_in_to_out(ica_input, ica_output)
    swapped_flag = False
    if abs(corrs[0]) > abs(corrs[1]):  # sources are not inverted
        ica_output_corrected = ica_output
    else:  # sources are inverted, reassign:
        swapped_flag = True
        a = copy.deepcopy(ica_output)
        b = copy.deepcopy(a[0, :])
        a[0, :] = a[1, :]
        a[1, :] = b
        ica_out_swapped = copy.deepcopy(a)
        ica_output_corrected = copy.deepcopy(ica_out_swapped)
    return ica_output_corrected, swapped_flag


def run_ica(ica_input: np.ndarray,
            iters: int,
            seed: int,
            verbose: bool = False) -> Union[Tuple[np.ndarray,
                                                  np.ndarray,
                                                  bool],
                                            Tuple[np.ndarray,
                                                  np.ndarray,
                                                  bool, bool]]:
    """
    ica_input -- an MxN numpy array; in the context of the decrosstalk
    problem, N=the number of timesteps; M=2 (first row is signal;
    second is crosstalk)

    iters -- an int; the number of iterative loops to try to go through
    to get the off-diagonal elements of the mixing matrix < 0.3

    seed -- an int; the seed of the random number generator that will
    be fed to sklearn.decompositions.FastICA

    verbose -- if True, also returns a flag indicating whether or not
    ICA output signals had to be swapped (just used for testing)

    Returns
    -------
    ica_output -- and MxN numpy array; ica_output[0,:] is the unmixed signal,
    ica_output[1,:] is the unmixed crosstalk (in the context of the decrosstalk
    problem)

    mixing -- the mixing matrix that gets from ica_input to ica_output
    np.dot(mixing, ica_output) will restore ica_input

    roi_demixed -- a boolean indicating whether or not the iteration to get
    the off-diagonal elements of the mixing matrix < 0.3 actually worked
    """

    # Whiten observations
    #
    # NOTE: we whiten the data by hand and then call
    # FastICA() with whiten=False to avoid running
    # afoul of this bug in sklearn
    #
    # https://github.com/scikit-learn/scikit-learn/issues/17162
    #
    # after this issue is resolved in sklearn, we can
    # revisit the possibility of using sklearn.FastICA's
    # internal whitening

    Ow, W, m = whiten_data(ica_input.transpose())
    alpha = 1
    beta = 1
    it = 0
    roi_demixed = False
    rng = np.random.RandomState(seed)

    while not roi_demixed and it <= iters:
        if alpha > 0.3 or beta > 0.3 or alpha < 0 or beta < 0:
            # Unmixing
            ica = FastICA(whiten=False, max_iter=10000, random_state=rng)
            ica.fit(Ow)  # Reconstruct sources
            mixing_raw = ica.mixing_

            # correcting for scale and offset:

            # applying inverse of whitening matrix
            M_hat = np.dot(np.linalg.inv(W), mixing_raw)

            # computing scaling matrix
            scale = np.dot(np.linalg.inv(M_hat), np.array([1, 1]))

            # applying scaling matrix
            mixing = M_hat * scale

        else:
            roi_demixed = True

        alpha = mixing[0, 1]
        beta = mixing[1, 0]
        it += 1

    # recovering outputs using new mixing matrix
    Sos = np.dot(np.linalg.inv(mixing), ica_input)

    # fixing source assignment ambiguity
    (ica_output,
     swapped) = fix_source_assignment(ica_input, Sos)

    if swapped:
        new_mixing = np.zeros((2, 2), dtype=float)
        new_mixing[:, 1] = mixing[:, 0]
        new_mixing[:, 0] = mixing[:, 1]
        mixing = new_mixing

    if verbose:
        return ica_output, mixing, roi_demixed, swapped

    return ica_output, mixing, roi_demixed
