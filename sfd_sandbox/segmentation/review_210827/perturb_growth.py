import matplotlib.figure as mplt_figure
import h5py
from ophys_etl.modules.segmentation.graph_utils.conversion import graph_to_img
from ophys_etl.modules.segmentation.detect.feature_vector_utils import (
    choose_timesteps,
    select_window_size)
from ophys_etl.modules.segmentation.detect.feature_vector_rois import (
    PearsonFeatureROI)

import pathlib
import numpy as np

import argparse

if __name__ == "__main__":
    exp_id = 785569447
    video_path = f'video/{exp_id}_denoised.h5'
    graph_path = f'graph/{exp_id}_graph.pkl'

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default=video_path)
    parser.add_argument('--graph', type=str, default=graph_path)
    parser.add_argument('--seed', type=int, default=725162)
    parser.add_argument('--rows', type=int, nargs='+', default=[140, 150])
    parser.add_argument('--cols', type=int, nargs='+', default=[50, 150])
    parser.add_argument('--plot_path', type=str, default=None)

    args = parser.parse_args()


    assert args.plot_path is not None

    video_path = pathlib.Path(args.video)
    graph_path = pathlib.Path(args.graph)


    #best_guess_rows = (250, 300)
    #best_guess_cols = (350, 450)

    #best_guess_rows = (140, 150)
    #best_guess_cols = (50, 150)


    best_guess_rows = tuple(args.rows)
    best_guess_cols = tuple(args.cols)

    full_img = graph_to_img(
                      graph_path,
                      attribute_name='filtered_hnc_Gaussian')

    rng = np.random.default_rng(args.seed)

    n_perturbations = 5
    fig = mplt_figure.Figure(figsize=(15,5*n_perturbations))
    axes = [fig.add_subplot(n_perturbations, 3, ii)
            for ii in range(1, 3*n_perturbations+1)]

    print('len axes ',len(axes))

    fontsize=15
    marker_size=100

    true_mask = None
    for i_fig in range(n_perturbations):
        print(i_fig*3)
        raw_axis = axes[i_fig*3]
        first_axis = axes[1+i_fig*3]
        last_axis = axes[2+i_fig*3]

        pixel_mask = np.zeros((512, 512), dtype=bool)

        sub_img = full_img[best_guess_rows[0]:best_guess_rows[1],
                           best_guess_cols[0]:best_guess_cols[1]]

        brightest = np.argmax(sub_img)
        sub_pt = np.unravel_index(brightest, sub_img.shape)
        seed_pt = (sub_pt[0]+best_guess_rows[0],
                   sub_pt[1]+best_guess_cols[0])


        if i_fig > 0:
            # mask pixels around seed_pt for timestep selection
            r = full_img.shape[0]
            c = full_img.shape[1]
            pixel_mask[max(0, seed_pt[0]-2):min(seed_pt[0]+2, r),
                       max(0, seed_pt[1]-2):min(seed_pt[1]+2, c)] = True

        window = select_window_size(
                      seed_pt,
                      full_img)

        r0 = int(max(0, seed_pt[0] - window))
        r1 = int(min(full_img.shape[0], seed_pt[0] + window))
        c0 = int(max(0, seed_pt[1] - window))
        c1 = int(min(full_img.shape[1], seed_pt[1] + window))

        window_img = full_img[r0:r1,
                              c0:c1]

        pixel_mask = pixel_mask[r0:r1,
                                c0:c1]

        if i_fig > 0:
            pixel_mask = rng.integers(0, 2, size=(r1-r0, c1-c0), dtype=bool)

        old_seed = seed_pt

        if i_fig > 0:
            valid_pixels = np.where(true_mask)
            pixel_indexes = np.arange(len(valid_pixels[0]))
            chosen_index = rng.choice(pixel_indexes)
            seed_pt = (valid_pixels[0][chosen_index]+r0,
                       valid_pixels[1][chosen_index]+c0)

        with h5py.File(video_path, 'r') as in_file:
            sub_video = in_file['data'][:, r0:r1, c0:c1]

        (timesteps,
         interesting_pts) = choose_timesteps(
                              sub_video,
                              (seed_pt[0]-r0, seed_pt[1]-c0),
                               0.2,
                               window_img,
                               pixel_ignore=pixel_mask)

        sub_video = sub_video[timesteps, :, :]

        roi = PearsonFeatureROI(
                  seed_pt,
                  (r0, c0),
                  sub_video)

        final_mask, diagnostic = roi.get_mask(3.0)

        img_max = window_img.max()
        img_min = window_img.min()
        d = img_max-img_min
        raw_img_rgb = np.round(255.0*(window_img.astype(float)-img_min)/d)
        assert raw_img_rgb.max() <= 255.0
        raw_img_rgb = raw_img_rgb.astype(np.uint8)
        img_rgb = np.zeros((raw_img_rgb.shape[0], raw_img_rgb.shape[1], 3),
                           dtype=np.uint8)
        for ic in range(3):
            img_rgb[:,:,ic] = raw_img_rgb

        n_iterations = len(diagnostic)

        n_graphs = n_iterations+1
        dim0 = np.ceil(np.sqrt(n_graphs)).astype(int)
        while (dim0)*dim0 < n_graphs:
            dim0 += 1

        nrows = dim0
        ncols = dim0

        raw_axis.imshow(img_rgb, zorder=0)
        raw_axis.set_title('window; seed and characteristic points',
                   fontsize=fontsize)
        raw_axis.scatter(seed_pt[1]-c0, seed_pt[0]-r0, color='r', zorder=1,
                         s=marker_size)
        raw_axis.scatter(old_seed[1]-c0, old_seed[0]-r0, marker='o',
                         s=marker_size, color='g', zorder=2)
        for pt in interesting_pts:
            raw_axis.scatter(pt[1], pt[0], color='r', marker='x', zorder=1,
                             s=marker_size)

        data = diagnostic[1]
        img = np.copy(img_rgb)
        #for mask in (data['background'], data['old'], data['new']):
        #    for ic in range(3):
        #        img[:,:,ic][mask] = 255

        if true_mask is None:
            true_mask = data['new']

        #img[:, :, 0][data['background']] = 125
        #img[:, :, 1][data['old']] = 125
        #img[:, :, 2][data['new']] = 125

        for k in ('old', 'new'):
            #img[:, :, 0][data[k]] = img[:, :, 0][data[k]]//4
            #img[:, :, 0][data[k]] += 190
            img[:, :, 0][data[k]] = 255
            img[:, :, 1][data[k]] = 0
            img[:, :, 2][data[k]] = 0

        first_axis.imshow(img)
        first_axis.set_title(f'first iteration', fontsize=fontsize)

        img = np.copy(img_rgb)
        img[:, :, 0][final_mask] = 255
        img[:, :, 1][final_mask] = 0
        img[:, :, 2][final_mask] = 0

        #img[:, :, 0][final_mask] += 125

        last_axis.imshow(img)
        last_axis.set_title(f'final ROI', fontsize=fontsize)


    fig.tight_layout()
    fig.savefig(args.plot_path, dpi=300)
