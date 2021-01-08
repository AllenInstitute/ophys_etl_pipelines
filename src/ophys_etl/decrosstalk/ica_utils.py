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


def run_ica(ica_input, iters, seed):
    # Whiten observations
    Ow, W, m = whiten_data(ica_input)
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

    Sos = np.dot(ica_input, np.linalg.inv(mixing).T)  # recovering outputs using new mixing matrix

    ica_output, corrs, _ = fix_source_assignment(ica_input.T, Sos.T) # fixing source assignment ambiguity

    return ica_output, mixing, roi_demixed
