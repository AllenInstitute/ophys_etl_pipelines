import logging
from types import ModuleType
from typing import List, Dict

import json

from ophys_etl.modules import segment_postprocess
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from sqlmodel import Session

from ophys_etl.workflows.ophys_experiment import OphysExperiment

from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum

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

        motion_corrected_ophys_movie_file: OutputFile = \
            kwargs['motion_corrected_ophys_movie_file']
        self._motion_corrected_ophys_movie_file = str(motion_corrected_ophys_movie_file.path)

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.TRACE_EXTRACTION

    @property
    def inputs(self, session: Session) -> Dict:
        return {
            'log_level': logging.DEBUG,
            'storage_directory': self.output_path,
            'motion_border': self.ophys_experiment.get_ophys_experiment_motion_border(
            session),
            'motion_corrected_stack': self._motion_corrected_ophys_movie_file,
            'rois': self.ophys_experiment.get_ophys_experiment_roi_metadata(
                                                      session),
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.TRACE_EXTRACTION_EXCLUSION_LABELS,
                path=self.output_metadata_path),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.ROI_TRACE,
                path=self.output_path),
             OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.NEUROPIL_TRACE,
                path=self.output_path),
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.NEUROPIL_MASK,
                path=self.output_path),
        ]
