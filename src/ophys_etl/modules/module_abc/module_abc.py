from typing import Union
import pathlib
import hashlib
import json
import h5py
import numpy as np
import argschema
from marshmallow import post_load
import pkg_resources
from abc import ABC, abstractmethod


class OphysEtlBaseSchema(argschema.ArgSchema):

    metadata_field = argschema.fields.String(
            default=None,
            required=True,
            allow_none=True,
            description=("Field point to file, either JSON or HDF5, "
                         "where metadata gets written"))

    @post_load
    def check_metadata_field(self, data, **kwargs):
        if data['metadata_field'] is None:
            return data

        if data['metadata_field'] not in data:
            msg = f"{data['metadata_field']} is not a field "
            msg += "in this schema"
            raise ValueError(msg)
        is_h5 = data[data['metadata_field']].endswith('.h5')
        is_json = data[data['metadata_field']].endswith('.json')
        if not is_h5 and not is_json:
            msg = f"metadata file {data[data['metadata_field']]} "
            msg += "is neither a .h5 or a .json"
            raise ValueError(msg)
        return data


def file_hash_from_path(file_path: Union[str, pathlib.Path]) -> str:
    """
    Return the hexadecimal file hash for a file

    Parameters
    ----------
    file_path: Union[str, Path]
        path to a file

    Returns
    -------
    str:
        The file hash (Blake2b; hexadecimal) of the file
    """
    hasher = hashlib.md5()
    with open(file_path, 'rb') as in_file:
        chunk = in_file.read(1000000)
        while len(chunk) > 0:
            hasher.update(chunk)
            chunk = in_file.read(1000000)
    return hasher.hexdigest()



def create_hashed_json(parameter_dict, to_skip=None):
    if to_skip is None:
        to_skip = set()
    output_list = list()
    key_list = list(parameter_dict.keys())
    key_list.sort()
    for key in key_list:
        value = parameter_dict[key]
        if isinstance(value, dict):
            output_list += create_hashed_json(parameter_dict[key],
                                              to_skip=to_skip)
        elif isinstance(value, str) or isinstance(value, pathlib.Path):
            file_path = pathlib.Path(value)
            if str(file_path.resolve().absolute()) in to_skip:
                continue
            local_dict = dict()
            local_dict['path'] = str(file_path.resolve().absolute())
            if file_path.is_file():
                file_hash = file_hash_from_path(file_path)
                local_dict['hash'] = file_hash
                output_list.append(local_dict)

    return output_list



def get_environment():
    package_names = []
    package_versions = []
    for p in pkg_resources.working_set:
        package_names.append(p.project_name)
        package_versions.append(p.version)
    package_names = np.array(package_names)
    package_versions = np.array(package_versions)
    sorted_dex = np.argsort(package_names)
    package_names = package_names[sorted_dex]
    package_versions = package_versions[sorted_dex]
    return [{'name':n, 'version': v}
            for n, v in zip(package_names, package_versions)]



class ModuleRunnerABC(ABC):

    @property
    def metadata_fname(self):
        if not hasattr(self, '_metadata_fname'):
            fname = pathlib.Path(self.args[self.args['metadata_field']])
            fname = str(fname.resolve().absolute())
            self._metadata_fname = fname
        return self._metadata_fname


    @abstractmethod
    def _run(self):
        raise NotImplementedError

    def run(self):
        self.output_metadata = dict()

        if self.args['metadata_field'] is not None:
            input_metadata = create_hashed_json(
                                    self.args,
                                    to_skip=set([self.metadata_fname]))

        self._run()

        if self.args['metadata_field'] is not None:
            output_paths = set([obj['path']
                                 for obj in self.output_metadata])
            n = len(input_metadata)
            for ii in range(n-1, -1, -1):
                if input_metadata[ii]['path'] in output_paths:
                    input_metadata.pop(ii)

            environ = get_environment()

            metadata = dict()
            metadata['environment'] = environ
            metadata['args'] = self.args
            metadata['input_files'] = input_metadata
            metadata['output_files'] = self.output_metadata

            metadata_fname = self.args[self.args['metadata_field']]
            if metadata_fname.endswith('h5'):
                with h5py.File(metadata_fname, 'a') as out_file:
                    assert 'metadata' not in out_file.keys()
                    out_file.create_dataset(
                            'metadata',
                            data=json.dumps(metadata).encode('utf-8'))
            elif metadata_fname.endswith('json'):
                with open(metadata_fname, 'rb') as in_file:
                    data = json.load(in_file)
                with open(metadata_fname, 'w') as out_file:
                    out_file.write(json.dumps({'metadata': metadata,
                                           'data': data}, indent=2))
            else:
                raise ValueError("Cannot handle metadata "
                                 f"file {metadata_fname}")


    def output(self, d, output_path=None, **json_dump_options):
        if self.args['metadata_field'] is not None:
            output_d = self.get_output_json(d)
            output_metadata = create_hashed_json(
                                    output_d,
                                    to_skip=self.metadata_fname)

        super().output(d, output_path=output_path, **json_dump_options)

        if self.args['metadata_field'] is not None:
            self.output_metadata = output_metadata

