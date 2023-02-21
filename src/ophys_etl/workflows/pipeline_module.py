"""Pipeline module"""
import abc
import os
from dataclasses import dataclass
from pathlib import Path

import json
from typing import Dict, List

from ophys_etl.workflows.utils.json_utils import EnhancedJSONEncoder

from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.well_known_file_types import WellKnownFileType


class ModuleOutputFileExistsError(Exception):
    pass


@dataclass
class OutputFile:
    """File output by module"""
    path: Path
    well_known_file_type: WellKnownFileType


class PipelineModule:
    """Pipeline module"""
    def __init__(
            self,
            ophys_experiment: OphysExperiment,
            debug: bool = False,
            prevent_file_overwrites: bool = True,
            docker_tag: str = 'main',
            **module_args
    ):
        """

        Parameters
        ----------
        ophys_experiment
            `OphysExperiment` instance
        debug
            Whether to run a dummy executable instead of actual one
        prevent_file_overwrites
            Whether to allow files output by module to be overwritten
        docker_tag
            What docker tag to use to run module
        """
        output_dir = ophys_experiment.output_dir / self.queue_name
        os.makedirs(output_dir, exist_ok=True)

        self._ophys_experiment = ophys_experiment
        self._debug = debug
        self._docker_tag = docker_tag

        if prevent_file_overwrites:
            self._validate_file_overwrite()

    @property
    @abc.abstractmethod
    def queue_name(self) -> str:
        """Identifier for 'queue' this module runs"""
        raise NotImplementedError

    @property
    def docker_tag(self) -> str:
        """What docker tag to use to run module"""
        return self._docker_tag

    @property
    def debug(self) -> bool:
        """Whether module is being run in debug mode"""
        return self._debug

    @property
    def ophys_experiment(self) -> OphysExperiment:
        """The `OphysExperiment` we are running the module on"""
        return self._ophys_experiment

    @property
    @abc.abstractmethod
    def inputs(self) -> Dict:
        """Input args to module"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def outputs(self) -> List[OutputFile]:
        """List of output files module outputs"""
        raise NotImplementedError

    @property
    def executable(self) -> str:
        """Fully qualified path to executable this module runs"""
        return 'sleep' if self._debug else self._executable

    @property
    @abc.abstractmethod
    def _executable(self) -> str:
        """Fully qualified path to executable this module runs"""
        raise NotImplementedError

    @property
    def executable_args(self) -> Dict:
        """Returns arguments to send to `executable`"""
        if self._debug:
            res = {
                'args': [60],    # i.e. will run sleep 60
                'kwargs': {}
            }
        else:
            res = {
                'args': [],
                'kwargs': {
                    'input_json': self.input_args_path,
                    'output_json': self.output_metadata_path
                }
            }
        return res

    @property
    def output_path(self) -> Path:
        """Where module is writing outputs to"""
        return self._ophys_experiment.output_dir / self.queue_name

    @property
    def output_metadata_path(self) -> Path:
        """Where to write output metadata to"""
        path = self.output_path / \
            f'{self.queue_name}_' \
            f'{self._ophys_experiment.id}_output.json'
        return path

    @property
    def input_args_path(self) -> Path:
        """Path to input arguments json file on disk"""
        args_path = self.output_path / \
            f'{self.queue_name}_' \
            f'{self._ophys_experiment.id}_input.json'
        return args_path

    def write_input_args(self):
        """Writes module input args to disk"""
        with open(self.input_args_path, 'w') as f:
            f.write(json.dumps(self.inputs, indent=2, cls=EnhancedJSONEncoder))

    def _validate_file_overwrite(
        self
    ):
        """
        Validates that that files which the module outputs don't exist

        Returns
        -------
        None

        Raises
        ------
        `ModuleOutputFileExistsError` if file exists
        """
        for out in self.outputs:
            if Path(out.path).exists():
                raise ModuleOutputFileExistsError(
                    f'{out.well_known_file_type} already exists at {out.path}')
