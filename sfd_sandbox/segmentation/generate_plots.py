import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import re
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
    output_dir = "/allen/aibs/informatics/danielsf/deep_interpolation"

    parent_dir="/allen/programs/braintv/workgroups/nc-ophys/danielk/deepinterpolation/experiments"

    id_pattern = re.compile('[0-9]+')
    flist = os.listdir(output_dir)

    for fname in flist:
        if not fname.endswith('pkl'):
            continue
        m = id_pattern.search(fname)
        exp_id = int(fname[m.start():m.end()])

        out_name = os.path.join(output_dir, f'ophys_exp_{exp_id}.png')
        sub_dir = os.path.join(parent_dir, f'ophys_experiment_{exp_id}')
        assert os.path.isdir(sub_dir)
        noisy_max = os.path.join(sub_dir, 'noised_maxp.png')
        denoised_max = os.path.join(sub_dir, 'denoised_maxp.png')
        new_graph = os.path.join(output_dir,
                                 fname)
        old_graph = os.path.join(sub_dir, 'graph_denoised.pkl')

        draw_figure(out_name=out_name,
            noisy_max=noisy_max,
            denoised_max=denoised_max,
            new_graph=new_graph,
            old_graph=old_graph)
