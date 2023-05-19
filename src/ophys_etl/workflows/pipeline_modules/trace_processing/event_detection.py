from types import ModuleType
from typing import List

from ophys_etl.modules import event_detection
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum


class EventDetection(PipelineModule):
    """Wrapper the event detection module"""

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

        dff_traces: OutputFile = kwargs[
            "dff_traces"
        ]
        self._dff_traces_file = str(
            dff_traces.path
        )

    @property
    def _executable(self) -> ModuleType:
        return event_detection

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.DEMIX_TRACES

    @property
    def inputs(self):
        roi_ids = []
        for roi in self.ophys_experiment.roi:
            if roi.is_valid:
                roi_ids.append(roi.id)

        module_args = {
            "movie_frame_rate_hz": self.ophys_experiment.movie_frame_rate_hz,
            "full_genotype": self.ophys_experiment.full_genotype,
            "ophysdfftracefile": self._dff_traces_file,
            "valid_roi_ids": roi_ids,
            "output_event_file": str(self._output_file_path)
        }
        return module_args

    @property
    def _output_file_path(self):
        return self.output_path \
               / f"{self._ophys_experiment.id}_event.h5"

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.EVENTS
                ),
                path=self._output_file_path,
            ),
        ]
