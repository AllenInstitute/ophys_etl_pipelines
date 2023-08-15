from types import ModuleType
from typing import List

from ophys_etl.workflows.app_config.app_config import app_config

from ophys_etl.modules import dff
from ophys_etl.modules.dff.schemas import DffJobSchema
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum


class DFOverFCalculation(PipelineModule):
    """Wrapper around df over f module"""

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

        neuropil_corrected_traces: OutputFile = kwargs[
            "neuropil_corrected_traces"
        ]
        self._neuropil_corrected_traces = str(neuropil_corrected_traces.path)

    @property
    def executable(self) -> ModuleType:
        return dff

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.DFF

    @property
    def inputs(self):
        module_args = {
            "input_file": self._neuropil_corrected_traces,
            "output_file": str(self._output_file_path),
            "movie_frame_rate_hz": self.ophys_experiment.movie_frame_rate_hz
        }
        if app_config.is_debug:
            # Making smaller due to short movie, otherwise it crashes
            module_args['long_baseline_filter_s'] = 6
        return DffJobSchema().load(data=module_args)

    @property
    def _output_file_path(self):
        return self.output_path \
               / f"{self._ophys_experiment.id}_dff.h5"

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.DFF_TRACES
                ),
                path=self._output_file_path,
            ),
        ]
