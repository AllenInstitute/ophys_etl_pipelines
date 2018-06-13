import os
import numpy as np
from .tiff import MesoscopeTiff


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


def scanfield_to_h5(h5fp, dset_name, scanfield_data, scanfield_metadata,
                    page_block_size=None, **h5_opts):
    if page_block_size is None:
        dset = h5fp.create_dataset(dset_name, data=scanfield_data[:],
                                   **h5_opts)
    else:
        dset = h5fp.create_dataset(dset_name, scanfield_data.shape,
                                   dtype=scanfield_data.dtype, **h5_opts)
        i = 0
        while i < scanfield_data.shape[0]:
            dset[i:i+page_block_size,:,:] = scanfield_data[i:i+page_block_size]
            i += page_block_size
    dump_dict_as_attrs(dset, None, scanfield_metadata)
