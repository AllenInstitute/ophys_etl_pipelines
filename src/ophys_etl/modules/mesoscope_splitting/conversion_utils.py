import numpy as np
from tifffile import imsave
from typing import Callable
import h5py

from ophys_etl.modules.mesoscope_splitting.tiff import DataView


def dump_dict_as_attrs(h5fp, container_name, data):
    if container_name is None:
        container = h5fp
    elif container_name in h5fp:
        raise AttributeError("{} already exists".format(container_name))
    else:
        container = h5fp.create_group(container_name)
    for key, value in data.items():
        if isinstance(value, dict):
            dump_dict_as_attrs(container, key, value)
        else:
            np_value = np.array(value)
            if np_value.dtype.type in (np.unicode_, np.object_):
                np_value = np_value.astype("S")
            container.attrs.create(key, data=np_value)
    return container


def volume_to_h5(h5fp: h5py.File,
                 volume: DataView,
                 dset_name: str = "data",
                 page_block_size: int = None):
    """Write a tiff volume to an HDF5 file.

    Parameters
    ----------
    h5fp : h5py.File
        An h5py.File handle, where the data will be saved.

    volume : DataView
        A DataView object as defined in the transforms.mesoscope_2p
        module. This contains the data to be saved.

    dset_name : str = "data"
        A string, this is what the dataset will be saved as in the .h5 file.

    page_block_size : int = None
        An optional integer used to save the data in chunks, if necessary.

    """
    h5_opts = {
            "chunks": (1,) + tuple(volume.plane_shape),
            "compression": 'gzip',
            "compression_opts": 4}
    if page_block_size is None:
        dset = h5fp.create_dataset(dset_name, data=volume[:], **h5_opts)
    else:
        dset = h5fp.create_dataset(dset_name, volume.shape,
                                   dtype=volume.dtype, **h5_opts)
        i = 0
        while i < volume.shape[0]:
            dset[i:i+page_block_size, :, :] = volume[i:i+page_block_size]
            i += page_block_size


def volume_to_tif(filename: str,
                  volume: DataView,
                  projection_func: Callable = None):
    """Write a volume to an HDF5 file.

    Parameters
    ----------
    Filename : str
        The filepath where the .tif will be saved.

    volume : DataView
        A DataView object as defined in the transforms.mesoscope_2p
        module. This contains the data to be saved.

    projection_func : Callable = None
        An optional function to pass the data through before saving.
        The only required argument for this function must be a numpy array.
    """
    if projection_func is not None:
        array = projection_func(volume[:])
        imsave(filename, array)
    else:
        imsave(filename, volume[:], bigtiff=True)


def average_and_unsign(volume):
    flat = np.mean(volume, axis=0)
    img32 = np.array(flat, dtype=np.uint32)
    max_intensity = np.max(flat)
    min_intensity = np.min(flat)

    return ((img32 - min_intensity)
            * 255.0
            / (max_intensity-min_intensity)).astype(np.uint8)
