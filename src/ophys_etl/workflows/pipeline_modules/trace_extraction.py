import logging
from types import ModuleType
from typing import List, Dict

import json

from ophys_etl.modules import segment_postprocess
from ophys_etl.workflows.workflow_steps import WorkflowStep
from sqlmodel import Session

from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValue
from ophys_etl.workflows.ophys_experiment import OphysExperiment

from ophys_etl.workflows.pipeline_module import PipelineModule, OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileType

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
    def queue_name(self) -> WorkflowStep:
        return WorkflowStep.TRACE_EXTRACTION

    @property
    def inputs(self) -> Dict:
        return {
            #TODO add inputs
            'log_level': logging.DEBUG,
            'storage_directory': self.output_path,
            'motion_border': {
                'y1': 0,
                'y0': 0,
                'x0': 0,
                'x1': 0,
            },
            'motion_corrected_stack': self._motion_corrected_ophys_movie_file
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=WellKnownFileType.NEUROPIL_TRACE,
                path=self.output_path),
             OutputFile(
                well_known_file_type=WellKnownFileType.NEUROPIL_TRACE,
                path=self.output_path)
        ]

