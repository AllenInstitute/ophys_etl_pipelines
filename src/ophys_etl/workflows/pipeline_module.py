"""Pipeline module"""
import abc
import datetime
import json
import logging
import os
from argschema.schemas import DefaultSchema
from pathlib import Path
from types import ModuleType
from typing import Dict, List, Optional

from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysSession, OphysContainer
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.utils.json_utils import EnhancedJSONEncoder
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class ModuleOutputFileExistsError(Exception):
    pass


logger = logging.getLogger(__name__)


class PipelineModule(abc.ABC):
    """Pipeline module

    This is an abstract base class for creating pipeline modules. When
    subclassing, developers should:
    - Implement the required abstract properties and methods.
    - Ensure any required instance-specific attributes are set before
      calling this class's `__init__`.

    See `DffCalculationModule` for an example subclass."""

    def __init__(
        self,
        docker_tag: str,
        ophys_experiment: Optional[OphysExperiment] = None,
        ophys_session: Optional[OphysSession] = None,
        ophys_container: Optional[OphysContainer] = None,
        prevent_file_overwrites: bool = True,
        **module_args,
    ):
        """
        NOTE: When subclassing, ensure any required instance-specific
        attributes are set before calling this method.

        Parameters
        ----------
        ophys_experiment
            `OphysExperiment` instance.
            If pipeline module does not run on a specific ophys experiment,
            this can be None
        ophys_session
            `OphysSession` instance.
            If this contains a value and `ophys_experiment`, `ophys_container`
            do not, we assume this module runs at the session level
        ophys_container
            `OphysContainer` instance.
            If this contains a value and `ophys_experiment`, `ophys_session`
            do not, we assume this module runs at the container level
        prevent_file_overwrites
            Whether to allow files output by module to be overwritten
        docker_tag
            What docker tag to use to run module.
        """

        self._ophys_experiment = ophys_experiment
        self._ophys_session = ophys_session
        self._ophys_container = ophys_container
        self._docker_tag = docker_tag
        self._now = datetime.datetime.now()
        os.makedirs(self.output_path, exist_ok=True)

        if prevent_file_overwrites:
            self._validate_file_overwrite()

        if isinstance(self.module_schema, DefaultSchema):
            self.validate_input_args()
        else:
            raise ValueError(
                f"module_schema must be subclass of DefaultSchema, "
                f"got {type(self.module_schema)}"
            )

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
    def ophys_container(self) -> Optional[OphysContainer]:
        """The `OphysContainer` we are running the module on.
        None if not running on a specific ophys container"""
        return self._ophys_container

    @property
    def python_interpreter_path(self) -> str:
        """Path to python interpreter"""
        return '/envs/ophys_etl/bin/python'

    @property
    def dockerhub_repository_name(self) -> str:
        return 'ophys_etl_pipelines'

    @property
    @abc.abstractmethod
    def module_schema(self) -> DefaultSchema:
        """Argschema to validate module_args"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def inputs(self) -> Dict:
        """Module args"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def outputs(self) -> List[OutputFile]:
        """List of output files module outputs"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def executable(self) -> ModuleType:
        """Fully qualified path to executable this module runs"""
        raise NotImplementedError

    @property
    def executable_args(self) -> Dict:
        """Returns arguments to send to `executable`"""
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
        elif self._ophys_container is not None:
            path = self._ophys_container.output_dir / self.queue_name.value
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

    def validate_input_args(self) -> None:
        """Validates module input args after it's been processed by
        EnhancedJSONEncoder"""
        encoded_json = json.dumps(self.inputs, cls=EnhancedJSONEncoder)
        preprocessed_inputs = json.loads(encoded_json)
        self.module_schema.load(data=preprocessed_inputs)

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
