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

    excluded = seed_h5_group["excluded_seeds"][()]
    zorder = 1  # start at one so markers are on top of background image
    if len(excluded) != 0:
        exclusion_reason = np.array(
                [i.decode("utf-8")
                 for i in seed_h5_group["exclusion_reason"][()]])
        unique_reasons = np.unique(exclusion_reason)

        # order exclusion reasons in descending order of "population"
        # so we plot the least numerous excluded pixels on top of the
        # most numerous excluded pixels
        ct_per_reason = np.array([len(np.where(exclusion_reason == i)[0])
                                  for i in unique_reasons])
        sorted_index = np.argsort(-1*ct_per_reason)
        unique_reasons = unique_reasons[sorted_index]
        for i_reason, reason in enumerate(unique_reasons):
            index = np.array([i == reason for i in exclusion_reason])

            axes.scatter(excluded[index, 1],
                         excluded[index, 0],
                         marker="x",
                         label=reason,
                         zorder=zorder,
                         alpha=0.5)
            zorder += 1

    # plot the provided seeds, using zorder to make sure
    # they end up on top of the excluded pixels
    provided = seed_h5_group["provided_seeds"][()]
    if len(provided) != 0:
        axes.scatter(provided[:, 1], provided[:, 0],
                     marker="o", label="provided seeds",
                     zorder=zorder)

    axes.legend(bbox_to_anchor=(1.05, 1))
    figure.tight_layout()
