import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
from numpy.fft import fft2, ifft2
import numpy as np
import networkx as nx
import re
import PIL

def isolate_low_frequency_modes(img, n_modes):
    transformed = fft2(img)

    ty = transformed.shape[0]
    tx = transformed.shape[1]

    fx = np.zeros(tx, dtype=int)
    fy = np.zeros(ty, dtype=int)
    for ii in range(tx):
        if ii<tx//2:
            fx[ii] = ii
        else:
            fx[ii] = tx-1-ii
    for ii in range(ty):
        if ii<ty//2:
            fy[ii] = ii
        else:
            fy[ii] = ty-1-ii

    freq_grid = np.meshgrid(fx, fy)

    freq_grid = freq_grid[0]**2+freq_grid[1]**2

    assert freq_grid.shape == transformed.shape

    freq_arr = np.unique(freq_grid.flatten())
    nf = len(freq_arr)

    mask = np.where(freq_grid>freq_arr[n_modes])
    transformed[mask] = 0.0
    new_img = ifft2(transformed)
    return new_img.real


def graph_to_img(graph_fname):
    graph = nx.read_gpickle(graph_fname)
    coords = np.array(list(graph.nodes))
    shape = tuple(coords.max(axis=0) + 1)
    img = np.zeros(shape)
    for node in graph.nodes:
        vals = [graph[node][i]["weight"] for i in graph.neighbors(node)]
        img[node[0], node[1]] = np.sum(vals)
    return img

def mask_img(img, frac):
    new_img = np.ones(img.shape)
    flux_arr = np.sort(img.flatten())
    ii = np.round(frac*len(flux_arr)).astype(int)
    mask = img<flux_arr[ii]
    new_img[mask] = 0
    return new_img


def _do_graph(graph_name, axes, suffix='', frac=0.9):
    img = graph_to_img(graph_name)

    axes[0].imshow(img)
    axes[0].set_title(f'image {suffix}', fontsize=10)

    lf_bckgd = isolate_low_frequency_modes(img, 10)
    #axes[1].imshow(lf_bckgd)
    #axes[1].set_title(f'low frequency background {suffix}', fontsize=10)

    diff = img-lf_bckgd
    axes[1].imshow(diff)
    axes[1].set_title(f'background subtracted {suffix}', fontsize=10)

    masked_img = mask_img(img, frac)
    axes[2].imshow(masked_img)
    axes[2].set_title(f'unsubtracted ROIs {suffix}')

    masked_img = mask_img(diff, frac)
    axes[3].imshow(masked_img)
    axes[3].set_title(f'subtracted ROIs {suffix}', fontsize=10)


def generate_figures(new_graph=None, old_graph=None,
                     noisy_max=None, denoised_max=None,
                     out_name=None, frac=0.9):

    fig, axes = plt.subplots(2,5,figsize=(25,10))
    for a in axes.flatten():
        a.tick_params(left=0,bottom=0,labelleft=0,labelbottom=0,
                      which='both', axis='both')
        for s in ('top', 'bottom', 'left', 'right'):
            a.spines[s].set_visible(False)

    noisy = PIL.Image.open(noisy_max, mode='r')
    nr = noisy.size[0]
    nc = noisy.size[1]
    noisy = np.array(noisy).reshape(nr, nc)
    axes[0][0].imshow(noisy, cmap='gray')
    axes[0][0].set_title('noisy max proj', fontsize=10)

    denoised = PIL.Image.open(denoised_max, mode='r')
    nr = denoised.size[0]
    nc = denoised.size[1]
    denoised = np.array(denoised).reshape(nr, nc)
    axes[1][0].imshow(denoised, cmap='gray')
    axes[1][0].set_title('denoised max proj', fontsize=10)

    _do_graph(old_graph, axes[0][1:], suffix = '(raw R)', frac=frac)
    _do_graph(new_graph, axes[1][1:], suffix='(filtered R)', frac=frac)

    for a in axes.flatten():
        for ix in range(100, 512, 100):
            a.axhline(ix, alpha=0.2, color='red')
            a.axvline(ix, alpha=0.2, color='red')

    fig.tight_layout()
    fig.savefig(out_name)


if __name__ == "__main__":

    output_dir = "/allen/aibs/informatics/danielsf/deep_interpolation"

    parent_dir="/allen/programs/braintv/workgroups/nc-ophys/danielk/deepinterpolation/experiments"

    id_pattern = re.compile('[0-9]+')
    flist = os.listdir(output_dir)

    frac=0.95

    for fname in flist:
        if not fname.endswith('pkl'):
            continue
        m = id_pattern.search(fname)
        exp_id = int(fname[m.start():m.end()])

        out_name = os.path.join(output_dir,
                                f'ROIS_ophys_exp_{exp_id}_{int(100*frac)}.png')
        sub_dir = os.path.join(parent_dir, f'ophys_experiment_{exp_id}')
        assert os.path.isdir(sub_dir)

        denoised_max = os.path.join(sub_dir, 'denoised_maxp.png')
        noisy_max = os.path.join(sub_dir, 'noised_maxp.png')


        new_graph = os.path.join(output_dir,
                                 fname)
        old_graph = os.path.join(sub_dir, 'graph_denoised.pkl')

        generate_figures(new_graph=new_graph, old_graph=old_graph,
                         noisy_max=noisy_max, denoised_max=denoised_max,
                         out_name=out_name, frac=frac)

        print('plotted ',out_name)

    #src_dir = '/Users/scott.daniel/Pika/deep_interpolation/ophys_experiment_794298187'
    #src_fname = os.path.join(src_dir, 'graph_794298187.pkl')

    #old_fname = os.path.join(src_dir, 'graph_denoised.pkl')
    #assert os.path.isfile(src_fname)

    #generate_figures(new_graph=src_fname,
    #                 old_graph=old_fname,
    #                 out_name='test_fft.png')
