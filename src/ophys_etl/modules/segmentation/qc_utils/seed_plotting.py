import h5py
import json

from ophys_etl.modules.segmentation.qc_utils.roi_utils import add_rois_to_axes


def plot_seeds_and_rois(axes, seed_h5_path, rois_path):
    with open(rois_path, "r") as f:
        rois = json.load(f)

    with h5py.File(seed_h5_path, "r") as f:
        seeds = f["seeds"]
        s_coords = seeds["coordinates"][()]
        s_excluded = seeds["excluded"][()]
        seed_img = f["seed_image"][()]

    axes.imshow(seed_img, cmap="gray")
    for c, e in zip(s_coords, s_excluded):
        marker = "o"
        if e:
            marker = "x"
        axes.plot(c[1], c[0], marker=marker, color="b")
    add_rois_to_axes(axes, rois, seed_img.shape)
