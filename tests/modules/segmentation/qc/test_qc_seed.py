import pytest
import h5py
import numpy as np
from matplotlib.figure import Figure

from ophys_etl.modules.segmentation.qc import seed as qcseed
from ophys_etl.modules.segmentation.seed import seeder


@pytest.fixture
def seed_h5_group(tmp_path):
    image = np.array([[0.0, 0.0, 0.4, 0.0],
                      [0.5, 0.0, 0.3, 0.0],
                      [0.0, 0.5, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0]])
    sb = seeder.ImageMetricSeeder(keep_fraction=1.0,
                                  seeder_grid_size=1)
    sb.select_seeds(image, sigma=None)
    # iterate through a few seeds, so smoe show as provided
    # and some show as excluded
    for i in range(5):
        next(sb)
    h5path = tmp_path / "seed_results.h5"
    with h5py.File(h5path, "w") as f:
        sb.log_to_h5_group(f)

    with h5py.File(h5path, "r") as f:
        yield f["seed"]


@pytest.mark.parametrize(
        "image_background",
        [None, np.zeros((512, 512)), "seed_image"])
def test_add_seeds_to_axes_default(seed_h5_group,  image_background):
    fig = Figure()
    axes = fig.add_subplot(111)
    qcseed.add_seeds_to_axes(figure=fig,
                             axes=axes,
                             seed_h5_group=seed_h5_group,
                             image_background=image_background)


def test_add_seeds_bad_image(seed_h5_group, tmp_path):
    junk = tmp_path / "an_image.png"
    fig = Figure()
    axes = fig.add_subplot(111)
    with pytest.raises(ValueError,
                       match=r"image background should be *"):
        qcseed.add_seeds_to_axes(figure=fig,
                                 axes=axes,
                                 seed_h5_group=seed_h5_group,
                                 image_background=junk)
