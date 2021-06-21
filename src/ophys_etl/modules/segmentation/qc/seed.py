import h5py
import numpy as np
from typing import Union, Optional
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def add_seeds_to_axes(
        figure: Figure,
        axes: Axes,
        seed_h5_group: h5py.Group,
        image_background: Optional[Union[np.ndarray, str]] = "seed_image"):
    """plots seeds optionally on top of the seeding image

    Parameters
    ----------
    axes: matplotlib.axes.Axes
        the axes to which to add the seeds
    seed_h5_group: h5py.Group
        the hdf5 group for seeding
    image_background: Optional[Union[np.ndarray, str]]
        the image or name of the image dataset over which to plot the seeds

    """
    # plot the background from file or directly
    if image_background is not None:
        if isinstance(image_background, np.ndarray):
            image = image_background
        elif isinstance(image_background, str):
            image = seed_h5_group[image_background][()]
        else:
            raise ValueError("image background should be of type "
                             f"np.ndarray|str not {type(image_background)}")
        axes.imshow(image, cmap="gray")

    # plot the seeds
    provided = seed_h5_group["provided_seeds"][()]
    if len(provided) != 0:
        axes.scatter(provided[:, 1], provided[:, 0],
                     marker="o", label="provided seeds")

    excluded = seed_h5_group["excluded_seeds"][()]
    if len(excluded) != 0:
        exclusion_reason = np.array(
                [i.decode("utf-8")
                 for i in seed_h5_group["exclusion_reason"][()]])
        reasons = np.unique(exclusion_reason)
        for reason in reasons:
            index = np.array([i == reason for i in exclusion_reason])
            axes.scatter(excluded[index, 1],
                         excluded[index, 0],
                         marker="x",
                         label=reason)

    axes.legend(bbox_to_anchor=(1.05, 1))
    figure.tight_layout()
