import numpy as np
import copy
from scipy.stats import pearsonr
from sklearn.decomposition import FastICA

__all__ = ["whiten_data",
           "fix_source_assignment",
           "pearsion_ica_in_to_out",
           "run_ica"]


def pearson_ica_in_to_out(signal_in, signal_out):
    """
    Function to compute correlations between two vectors (traces in our case)
    :param signal_in:
    :param signal_out:
    :return:
    """
    assert signal_in.shape == signal_out.shape, "Shape of inputs doesn't align"
    assert signal_in.shape[0] < signal_in.shape[1], f"Shape of input is reversed : {signal_in.shape}, use input.T"
    assert signal_out.shape[0] < signal_out.shape[1], f"Shape of input is reversed : {signal_in.shape}, use input.T"
    cor_0_0, _ = pearsonr(signal_out[0, :], signal_in[0, :])
    cor_0_1, _ = pearsonr(signal_out[0, :], signal_in[1, :])
    cor_1_0, _ = pearsonr(signal_out[1, :], signal_in[0, :])
    cor_1_1, _ = pearsonr(signal_out[1, :], signal_in[1, :])
    return [cor_0_0, cor_0_1, cor_1_1, cor_1_0]


def whiten_data(x):
    """
    Function to debias (subtract mean) and whiten the data
    :param x:  -- shaped NxM
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


def fix_source_assignment(ica_input, ica_output):
    """
    Function to fix source assignment ambiguity of ICA
    :param ica_input:
    :param ica_output:
    :return:
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
    return ica_output_corrected, corrs, swapped_flag


def run_ica(ica_input, iters, seed, verbose=False):
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
    Ow, W, m = whiten_data(ica_input.transpose())
    alpha = 1
    beta = 1
    it = 0
    roi_demixed = False
    rng = np.random.RandomState(seed)

    while not roi_demixed and it <= iters:
        if alpha > 0.3 or beta > 0.3 or alpha < 0 or beta < 0:
            ### Unmixing
            ica = FastICA(whiten=False, max_iter=10000, random_state=rng)
            ica.fit_transform(Ow)  # Reconstruct sources
            mixing_raw = ica.mixing_

            # correcting for scale and offset:
            M_hat = np.dot(np.linalg.inv(W), mixing_raw)  # applying inverse of whitening matrix
            scale = np.dot(np.linalg.inv(M_hat), np.array([1, 1]))  # computing scaling matrix
            mixing = M_hat * scale # applying scaling matrix

        else:
            roi_demixed = True

        alpha = mixing[0, 1]
        beta = mixing[1, 0]
        it += 1

    Sos = np.dot(np.linalg.inv(mixing), ica_input)  # recovering outputs using new mixing matrix

    ica_output, corrs, swapped = fix_source_assignment(ica_input, Sos) # fixing source assignment ambiguity

    if swapped:
        new_mixing = np.zeros((2,2), dtype=float)
        new_mixing[:,1] = mixing[:,0]
        new_mixing[:,0] = mixing[:,1]
        mixing = new_mixing

    if verbose:
        return ica_output, mixing, roi_demixed, swapped

    return ica_output, mixing, roi_demixed
