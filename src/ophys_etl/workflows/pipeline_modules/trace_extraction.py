import logging
from types import ModuleType
from typing import List, Dict

import json

from ophys_etl.modules import segment_postprocess
from ophys_etl.workflows.workflow_steps import WorkflowStep
from sqlmodel import Session

from ophys_etl.workflows.db.schemas import OphysROI, OphysROIMaskValueDB
from ophys_etl.workflows.db.db_utils import get_ophys_experiment_roi_metadata, \
get_ophys_experiment_motion_border
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
    def inputs(self, session: Session) -> Dict:
        return {
            #TODO add inputs
            'log_level': logging.DEBUG,
            'storage_directory': self.output_path,
            'motion_border': get_ophys_experiment_motion_border(
            self.ophys_experiment.id, session),
            'motion_corrected_stack': self._motion_corrected_ophys_movie_file,
            'rois': get_ophys_experiment_roi_metadata(self.ophys_experiment.id,
                                                      session),
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=WellKnownFileType.TRACE_EXTRACTION_EXCLUSION_LABELS,
                path=self.output_metadata_path),
            OutputFile(
                well_known_file_type=WellKnownFileType.ROI_TRACE,
                path=self.output_path),
             OutputFile(
                well_known_file_type=WellKnownFileType.NEUROPIL_TRACE,
                path=self.output_path)
        ]
