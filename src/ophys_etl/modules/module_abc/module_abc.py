from typing import Union
from pathlib import Path
import hashlib
import json
import numpy as np
import pkg_resources
from abc import ABC, abstractmethod


def file_hash_from_path(file_path: Union[str, Path]) -> str:
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



def create_hashed_json(parameter_dict):
    output_list = list()
    key_list = list(parameter_dict.keys())
    key_list.sort()
    for key in key_list:
        value = parameter_dict[key]
        if isinstance(value, dict):
            output_list += create_hashed_json(parameter_dict[key])
        elif isinstance(value, str) or isinstance(value, Path):
            file_path = Path(value)
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

    @abstractmethod
    def _run(self):
        raise NotImplementedError

    def run(self):
        self.output_metadata = dict()
        print("SFD calling ABC.run")
        input_metadata = create_hashed_json(self.args)
        #print('input_metadata')
        #print(json.dumps(input_metadata, indent=2))
        self._run()

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
        print(json.dumps(metadata, indent=2))


    def output(self, d, output_path=None, **json_dump_options):
        output_d = self.get_output_json(d)
        output_metadata = create_hashed_json(d)
        #print('output_metadata')
        #print(json.dumps(output_metadata, indent=2))
        super().output(d, output_path=output_path, **json_dump_options)
        self.output_metadata = output_metadata

