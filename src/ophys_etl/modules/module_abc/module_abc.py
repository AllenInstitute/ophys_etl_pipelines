from __future__ import annotations

from typing import Union, Optional
import pathlib
import hashlib
import json
import h5py
import numpy as np
import copy
import argschema
from marshmallow import post_load
import pkg_resources
from abc import ABC, abstractmethod

class WorkflowElement(object):
    """
    A class to represent a single element in a workflow.
    It will keep track of the element that called it,
    the elements that it calls, and any metadata (parameters,
    input files, output files) associated with the current
    module.

    Parameters
    ----------
    name: str
        Representation of the name of the module
        class calling this element

    salt: Optional[str]
        Optional suffix to be added to this element's
        name to make it unique (in case the same module
        is called more than once in a workflow)

    called_by: Optional[WorkflowElement]
        The element that called this element
    """

    def __init__(self,
                 name: str,
                 salt: Optional[str],
                 called_by: Optional[WorkflowElement]):
        self.called_by = called_by
        if salt is not None:
            self.name = f'{name}_{salt}'
        else:
            self.name = str(name)
        if called_by is not None:
            self.called_by = called_by.name
        else:
            self.called_by = None
        self.called = None
        self.metadata = dict()

    def call(self, other: WorkflowElement):
        """
        Record that the current WorkflowElement is calling
        another WorkflowElement
        """
        if self.called is None:
            self.called = list()
        self.called.append(other.name)

    def add_metadata(self, metadata: dict):
        """
        Add a dict to the metadata for this WorkflowElement
        """
        self.metadata.update(metadata)

    def as_dict(self) -> dict:
        """
        Return a dict representation of this WorkflowElement
        """
        value = dict()
        value['name'] = self.name
        value['called_by'] = self.called_by
        if self.called is not None:
            value['called'] = self.called
        if len(self.metadata) > 0:
            value['metadata'] = self.metadata

        return value


class WorkflowLogger(object):
    """
    A class to keep track of a series of sequential
    workflow elements.

    Will not be able to handle workflow elements that run in parallel,
    but can keep track of elements that call other elements that
    themselves call other elements.
    """

    def __init__(self):
        self.log = list()
        self.currently_in = list()
        self.name_to_process = dict()


    def enter(self, name: str) -> str:
        """
        Enter a processing module.Call this method before
        executing the module's logic.

        Parameters
        ----------
        name: str
            A string representation of the name of the
            processing module class we are entering

        Returns
        -------
        new_name: str
            String representation of the module name as it
            was actually recorded in the log (in the event
            that the same module is called more than once
            in the workflow, a unique 'salt' is added to the
            name to prevent collisions in the workflow.
        """
        if str(name) not in self.name_to_process:
            salt = None
        else:
            new_name = name
            salt = 1
            while new_name in self.name_to_process:
                new_name = f'{name}_{salt}'
                salt += 1

        if len(self.currently_in) == 0:
            called_by = None
        else:
            called_by = self.currently_in[-1]
        new_process = WorkflowElement(name, salt, called_by)

        if called_by is not None:
            self.name_to_process[called_by.name].call(new_process)
        self.log.append(new_process)
        self.currently_in.append(new_process)
        self.name_to_process[new_process.name] = new_process
        return new_process.name


    def add_metadata(self, process_name: str, metadata: dict) -> None:
        """
        Add metadata to a recorded wordflow element.

        Parameters
        ----------
        process_name: str
            The name of the workflow element as it was recorded
            in the log (i.e. as returned by self.enter())

t        metadata: dict
            The metadata to be added to the element's metadata
        """
        self.name_to_process[process_name].add_metadata(metadata)
        return None


    def leave(self) -> None:
        """
        Leave the workflow element. Call this method after
        executing the module's logic
        """
        if len(self.currently_in) > 0:
            self.currently_in.pop(-1)
        return None

    def return_log(self) -> list:
        """
        Return the contents of this workflow log as a list
        """
        output = list()
        for process in self.log:
            output.append(process.as_dict())
        return output



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
    flow_logger = WorkflowLogger()

    @property
    def metadata_fname(self):
        if self.args['metadata_field'] is None:
            return None

        if not hasattr(self, '_metadata_fname'):
            fname = pathlib.Path(self.args[self.args['metadata_field']])
            fname = str(fname.resolve().absolute())
            self._metadata_fname = fname

        return self._metadata_fname


    @abstractmethod
    def _run(self):
        raise NotImplementedError

    def run(self):
        this_process_name = self.flow_logger.enter(type(self))

        self.output_metadata = dict()

        input_metadata = create_hashed_json(
                                self.args,
                                to_skip=set([self.metadata_fname]))

        self._run()
        self.flow_logger.leave()

        output_paths = set([obj['path']
                            for obj in self.output_metadata])
        n = len(input_metadata)
        for ii in range(n-1, -1, -1):
            if input_metadata[ii]['path'] in output_paths:
                input_metadata.pop(ii)

        metadata = dict()
        metadata['args'] = self.args
        metadata['input_files'] = input_metadata
        metadata['output_files'] = self.output_metadata
        self.flow_logger.add_metadata(this_process_name, metadata)

        if self.args['metadata_field'] is not None:
            output_metadata = dict()
            output_metadata['workflow'] = self.flow_logger.return_log()
            environ = get_environment()
            output_metadata['environment'] = environ
            metadata_fname = self.args[self.args['metadata_field']]
            if metadata_fname.endswith('h5'):
                with h5py.File(metadata_fname, 'a') as out_file:
                    assert 'metadata' not in out_file.keys()
                    out_file.create_dataset(
                            'metadata',
                            data=json.dumps(output_metadata).encode('utf-8'))
            elif metadata_fname.endswith('json'):
                with open(metadata_fname, 'rb') as in_file:
                    data = json.load(in_file)
                with open(metadata_fname, 'w') as out_file:
                    out_file.write(json.dumps({'metadata': output_metadata,
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

