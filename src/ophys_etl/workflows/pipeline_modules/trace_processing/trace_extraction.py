from types import ModuleType
from typing import Dict, List

from ophys_etl.modules import trace_extraction
from ophys_etl.modules.trace_extraction.schemas import TraceExtractionInputSchema  # noqa: E501
from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class TraceExtractionModule(PipelineModule):
    """Trace extraction module"""

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

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.TRACE_EXTRACTION

    @property
    def module_schema(self) -> TraceExtractionInputSchema:
        return TraceExtractionInputSchema()

    @property
    def inputs(self) -> Dict:
        return {
            "storage_directory": self.output_path,
            "motion_border": self.ophys_experiment.motion_border.to_dict(),
            "motion_corrected_stack": (
                self._motion_corrected_ophys_movie_file),
            "rois": [x.to_dict() for x in
                     self.ophys_experiment.rois]
                     }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.TRACE_EXTRACTION_EXCLUSION_LABELS, # noqa E501
                path=self.output_metadata_path,
            ),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.ROI_TRACE,
                path=self.output_path / 'roi_traces.h5',
            ),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.NEUROPIL_TRACE,
                path=self.output_path / 'neuropil_traces.h5',
            ),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.NEUROPIL_MASK,
                path=self.output_path / 'neuropil_masks.json',
            ),
        ]

    @property
    def executable(self) -> ModuleType:
        return trace_extraction
