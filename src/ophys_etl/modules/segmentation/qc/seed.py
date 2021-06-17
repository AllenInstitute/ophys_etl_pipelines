import h5py
import numpy as np
from pathlib import Path
from typing import Union, Optional
from matplotlib.axes import Axes


def add_seeds_to_axes(
        axes: Axes,
        seed_h5_group: Union[str, h5py.Group] = "seeding",
        image_background: Optional[Union[np.ndarray, str]] = "seed_image",
        seed_h5_path: Optional[Union[str, Path]] = None):
    """plots seeds optionally on top of the seeding image

    Parameters
    ----------
    axes: matplotlib.axes.Axes
        the axes to which to add the seeds
    seed_h5_group: Union[str, h5py.Group]
        the hdf5 group or name of the hdf5 group
    image_background: Optional[Union[np.ndarray, str]]
        the image or name of the image dataset over which to plot the seeds
    seed_h5_path: Optional[Union[str, Path]]
        the path to the hdf5 file

    """
    if (seed_h5_path is not None) & (not isinstance(seed_h5_group, str)):
        raise ValueError("for (seed_h5_path, seed_h5_group) provide either "
                         "types (Path|str, str) or (NoneType, h5py.Group). "
                         f"You provided ({type(seed_h5_path)}, "
                         f"{type(seed_h5_group)}).")

    # get the h5py.Group from file or directly
    if isinstance(seed_h5_path, Path) | isinstance(seed_h5_path, str):
        h5file = h5py.File(seed_h5_path, "r")
        group = h5file[seed_h5_group]
    else:
        group = seed_h5_group

    # plot the background from file or directly
    if image_background is not None:
        if isinstance(image_background, np.ndarray):
            image = image_background
        elif isinstance(image_background, str):
            image = group[image_background][()]
        else:
            raise ValueError("image background should be of type "
                             f"np.ndarray|str not {type(image_background)}")
        axes.imshow(image, cmap="gray")

    # plot the seeds
    provided = group["provided_seeds"][()]
    print(provided)
    if len(provided) != 0:
        axes.scatter(provided[:, 0], provided[:, 1],
                     marker="o", label="provided seeds")

    excluded = group["excluded_seeds"][()]
    if len(excluded) != 0:
        exclusion_reason = np.array([i.decode("utf-8")
                                     for i in group["exclusion_reason"][()]])
        reasons, indices = np.unique(exclusion_reason, return_inverse=True)
        for reason in reasons:
            index = indices == reason
            axes.scatter(excluded[index, 0],
                         excluded[index, 1],
                         marker="x",
                         label=reason)

    axes.legend()

    if isinstance(seed_h5_path, Path) | isinstance(seed_h5_path, str):
        h5file.close()
