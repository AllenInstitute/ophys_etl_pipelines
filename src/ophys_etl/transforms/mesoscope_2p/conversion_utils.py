import numpy as np
from tifffile import imsave
import h5py


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


def volume_to_h5(h5fp, volume, dset_name="data", page_block_size=None,
                 **h5_opts):
    if isinstance(h5fp, str):
        f = h5py.File(h5fp, "w")

    if page_block_size is None:
        dset = f.create_dataset(dset_name, data=volume[:],
                                **h5_opts)
    else:
        dset = f.create_dataset(dset_name, volume.shape,
                                dtype=volume.dtype, **h5_opts)
        i = 0
        while i < volume.shape[0]:
            dset[i:i+page_block_size, :, :] = volume[i:i+page_block_size]
            i += page_block_size


def volume_to_tif(filename, volume, projection_func=None):
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
