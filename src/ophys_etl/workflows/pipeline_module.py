"""Pipeline module"""
import abc
import datetime
import json
import logging
import os
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.ophys_experiment import OphysExperiment, OphysSession
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.utils.json_utils import EnhancedJSONEncoder
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class ModuleOutputFileExistsError(Exception):
    pass


logger = logging.getLogger(__name__)


class PipelineModule:
    """Pipeline module"""

    def __init__(
        self,
        docker_tag: str,
        ophys_experiment: Optional[OphysExperiment] = None,
        ophys_session: Optional[OphysSession] = None,
        prevent_file_overwrites: bool = True,
        **module_args,
    ):
        """

        Parameters
        ----------
        ophys_experiment
            `OphysExperiment` instance.
            If pipeline module does not run on a specific ophys experiment,
            this can be None
        ophys_session
            `OphysSession` instance.
            If this contains a value and `ophys_experiment` does not,
            we assume this module runs at the session level
        prevent_file_overwrites
            Whether to allow files output by module to be overwritten
        docker_tag
            What docker tag to use to run module.
        """

        self._ophys_experiment = ophys_experiment
        self._ophys_session = ophys_session
        self._docker_tag = docker_tag
        self._now = datetime.datetime.now()

        os.makedirs(self.output_path, exist_ok=True)

        if prevent_file_overwrites:
            self._validate_file_overwrite()

    @property
    @abc.abstractmethod
    def queue_name(self) -> WorkflowStepEnum:
        """Identifier for 'queue' this module runs"""
        raise NotImplementedError

    @property
    def docker_tag(self) -> str:
        """What docker tag to use to run module"""
        return self._docker_tag

    @property
    def ophys_experiment(self) -> Optional[OphysExperiment]:
        """The `OphysExperiment` we are running the module on.
        None if not running on a specific ophys experiment"""
        return self._ophys_experiment

    @property
    def ophys_session(self) -> Optional[OphysSession]:
        """The `OphysSession` we are running the module on.
        None if not running on a specific ophys session"""
        return self._ophys_session

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
        return "sleep" if app_config.is_debug else self._executable.__name__

    @property
    @abc.abstractmethod
    def _executable(self) -> ModuleType:
        """Module to run"""
        raise NotImplementedError

    @property
    def executable_args(self) -> Dict:
        """Returns arguments to send to `executable`"""
        if app_config.is_debug:
            res = {"args": [60], "kwargs": {}}  # i.e. will run sleep 60
        else:
            res = {
                "args": [],
                "kwargs": {
                    "input_json": self.input_args_path,
                    "output_json": self.output_metadata_path,
                },
            }
        return res

    @property
    def output_path(self) -> Path:
        """Where module is writing outputs to"""
        if self._ophys_session is not None:
            path = self._ophys_session.output_dir / self.queue_name.value
        elif self._ophys_experiment is not None:
            path = self._ophys_experiment.output_dir / self.queue_name.value
        else:
            path = app_config.output_dir / self.queue_name.value

        path = path / self.now_str

        return path

    @property
    def now_str(self) -> str:
        return self._now.strftime("%Y-%m-%d_%H-%m-%S-%f")

    @property
    def output_metadata_path(self) -> Path:
        """Where to write output metadata to"""
        path = self.output_path / f'{self.queue_name.value}_output.json'
        return path

    @property
    def input_args_path(self) -> Path:
        """Path to input arguments json file on disk"""
        args_path = (
            self.output_path / f"{self.queue_name.value}_input.json"
        )
        return args_path

    def write_input_args(self):
        """Writes module input args to disk"""
        with open(self.input_args_path, "w") as f:
            f.write(json.dumps(self.inputs, indent=2, cls=EnhancedJSONEncoder))

    def _validate_file_overwrite(self):
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
                    f"{out.well_known_file_type} already exists at {out.path}"
                )
