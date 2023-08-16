from types import ModuleType
from typing import List, Dict

from ophys_etl.modules import neuropil_correction
from ophys_etl.modules.neuropil_correction.schemas import NeuropilCorrectionJobSchema # noqa: E501
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
        return WorkflowStepEnum.NEUROPIL_CORRECTION

    @property
    def module_argschema(self) -> NeuropilCorrectionJobSchema:
        return NeuropilCorrectionJobSchema()

    @property
    def module_args(self) -> Dict:
        return {
            "roi_trace_file": self._demixed_roi_traces_file,
            "storage_directory": str(self.output_path),
            "neuropil_trace_file": str(self._neuropil_traces_file)
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.NEUROPIL_CORRECTED_TRACES
                ),
                path=self.output_path / "neuropil_correction.h5"
            ),
        ]
