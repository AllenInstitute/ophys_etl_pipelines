from types import ModuleType
from typing import List

from ophys_etl.modules import event_detection
from ophys_etl.modules.event_detection.schemas import EventDetectionInputSchema
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum


class EventDetectionModule(PipelineModule):
    """Wrapper the event detection module"""

    def __init__(
        self,
        ophys_experiment: OphysExperiment,
        prevent_file_overwrites: bool = True,
        **kwargs
    ):
        dff_traces: OutputFile = kwargs[
            "dff_traces"
        ]
        self._dff_traces_file = str(
            dff_traces.path
        )

        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs
        )

    @property
    def executable(self) -> ModuleType:
        return event_detection

    @property
    def python_interpreter_path(self) -> str:
        return '/envs/event_detection/bin/python'

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.EVENT_DETECTION

    @property
    def module_schema(self) -> EventDetectionInputSchema:
        return EventDetectionInputSchema()

    @property
    def module_args(self):
        valid_roi_ids = [roi.id for roi in self.ophys_experiment.rois if
                         roi.is_valid(self.ophys_experiment.equipment_name)]
        return {
                "movie_frame_rate_hz": self.ophys_experiment.movie_frame_rate_hz,  # noqa 501
                "full_genotype": self.ophys_experiment.full_genotype,
                "ophysdfftracefile": self._dff_traces_file,
                "valid_roi_ids": valid_roi_ids,
                "output_event_file": str(self._output_file_path)
            }

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
                path=self._output_file_path
            )
        ]
