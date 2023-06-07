from types import ModuleType
from typing import List

from ophys_etl.modules import neuropil_correction
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum


class NeuropilCorrection(PipelineModule):
    """Wrapper around neuropil correction module"""

    def __init__(
        self,
        ophys_experiment: OphysExperiment,
        prevent_file_overwrites: bool = True,
        **kwargs
    ):
        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs
        )

        motion_corrected_ophys_movie_file: OutputFile = kwargs[
            "motion_corrected_ophys_movie_file"
        ]
        self._motion_corrected_ophys_movie_file = str(
            motion_corrected_ophys_movie_file.path
        )
        demixed_roi_traces_file: OutputFile = kwargs[
            "demixed_roi_traces_file"
        ]
        self._demixed_roi_traces_file = str(demixed_roi_traces_file.path)
        neuropil_traces_file: OutputFile = kwargs[
            "neuropil_traces_file"
        ]
        self._neuropil_traces_file = str(neuropil_traces_file.path)

    @property
    def executable(self) -> ModuleType:
        return neuropil_correction

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.NEUROPIL_SUBTRACTION

    @property
    def inputs(self):
        module_args = {
            "motion_corrected_stack": self._motion_corrected_ophys_movie_file,
            "roi_trace_file": self._demixed_roi_traces_file,
            "storage_directory": str(self.output_path),
            "neuropil_trace_file": str(self._neuropil_traces_file)
        }
        return module_args

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.NEUROPIL_CORRECTED_TRACES
                ),
                path=(
                    self.output_path
                    / f"{self._ophys_experiment.id}_neuropil_correction.h5"
                ),
            ),
        ]
