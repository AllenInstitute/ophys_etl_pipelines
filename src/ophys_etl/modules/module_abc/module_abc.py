from typing import Union
from pathlib import Path
import hashlib
import json
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
    output_dict = dict()
    key_list = list(parameter_dict.keys())
    key_list.sort()
    for key in key_list:
        value = parameter_dict[key]
        if isinstance(value, dict):
            output_dict.update(create_hashed_json(parameter_dict[key]))
        elif isinstance(value, str) or isinstance(value, Path):
            file_path = Path(value)
            local_dict = dict()
            local_dict['path'] = str(file_path.resolve().absolute())
            if file_path.is_file():
                file_hash = file_hash_from_path(file_path)
                local_dict['hash'] = file_hash
                output_dict[key] = local_dict

    return output_dict



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
        for k in self.output_metadata:
            if k in input_metadata:
                input_metadata.pop(k)
        print('input_metadata')
        print(json.dumps(input_metadata, indent=2))
        print('output_metadata')
        print(json.dumps(self.output_metadata, indent=2))

    def output(self, d, output_path=None, **json_dump_options):
        output_d = self.get_output_json(d)
        output_metadata = create_hashed_json(d)
        #print('output_metadata')
        #print(json.dumps(output_metadata, indent=2))
        super().output(d, output_path=output_path, **json_dump_options)
        self.output_metadata = output_metadata
        
