import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import networkx as nx
import PIL
import numpy as np
import os

from ophys_etl.qc.video.correlation_graph_plot import draw_graph_edges

def draw_figure(old_graph=None,
                new_graph=None,
                noisy_max=None,
                denoised_max=None,
                out_name=None):

    fig, axes = plt.subplots(2,2, figsize=(10,10))
    for a in axes.flatten():
        a.tick_params(axis='both', which='both', left=0, bottom=0,
                      labelleft=0, labelbottom=0)

        for s in ('top', 'bottom', 'left', 'right'):
            a.spines[s].set_visible(False)

    noisy = PIL.Image.open(noisy_max, mode='r')
    nr = noisy.size[0]
    nc = noisy.size[1]
    noisy = np.array(noisy).reshape(nr, nc)
    axes[0][0].imshow(noisy, cmap='gray')
    axes[0][0].set_title('noisy max', fontsize=10)

    denoised = PIL.Image.open(denoised_max, mode='r')
    nr = denoised.size[0]
    nc = denoised.size[1]
    denoised = np.array(denoised).reshape(nr, nc)
    axes[0][1].imshow(denoised, cmap='gray')
    axes[0][1].set_title('denoised max', fontsize=10)

    old = nx.read_gpickle(old_graph)
    draw_graph_edges(fig, axes[1][0], old)
    axes[1][0].set_title('Pearson coeff', fontsize=10)
    del old
    new = nx.read_gpickle(new_graph)
    draw_graph_edges(fig, axes[1][1], new)
    axes[1][1].set_title('filtered Pearson coeff', fontsize=10)

    for a in axes.flatten():
        for val in range(100, 512, 100):
            a.axhline(val, alpha=0.25)
            a.axvline(val, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_name)

if __name__ == "__main__":

    data_dir = '/Users/scott.daniel/Pika/deep_interpolation/data'
    dummy_data_dir = '/Users/scott.daniel/Pika/deep_interpolation/dummy_data'
    ophys_dir = '/Users/scott.daniel/Pika/deep_interpolation/ophys_experiment_794298187'

    draw_figure(out_name='ophys_exp_1048483611.png',
        noisy_max=os.path.join(data_dir, 'noised_maxp.png'),
        denoised_max=os.path.join(data_dir, 'denoised_maxp.png'),
        new_graph=os.path.join(dummy_data_dir, 'graph_full.pkl'),
        old_graph=os.path.join(data_dir, 'graph_denoised.pkl'))

    draw_figure(out_name='ophys_exp_794298187.png',
        noisy_max=os.path.join(ophys_dir, 'noised_maxp.png'),
        denoised_max=os.path.join(ophys_dir, 'denoised_maxp.png'),
        new_graph=os.path.join(ophys_dir, 'graph_794298187.pkl'),
        old_graph=os.path.join(ophys_dir, 'graph_denoised.pkl'))
