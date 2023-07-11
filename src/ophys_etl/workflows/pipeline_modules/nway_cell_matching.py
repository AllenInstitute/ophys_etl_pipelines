from types import ModuleType
from typing import List, Dict

import json
from nway import nway_matching
from nway.schemas import OnPremGeneratedInputSchema
from sqlmodel import Session

from ophys_etl.workflows.db.schemas import NwayCellMatch
from ophys_etl.workflows.workflow_names import WorkflowNameEnum

from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum

from ophys_etl.workflows.db import engine

from ophys_etl.workflows.workflow_step_runs import \
    get_well_known_file_for_latest_run

from ophys_etl.workflows.ophys_experiment import OphysExperiment
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class NwayCellMatchingModule(PipelineModule):
    """Nway cell matching module"""

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.NWAY_CELL_MATCHING

    @property
    def inputs(self) -> Dict:
        return OnPremGeneratedInputSchema().load(data={
            'output_directory': str(self.output_path),
            'experiment_containers': {
                'ophys_experiments': self._get_container_experiments_input()
            }
        })

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.NWAY_CELL_MATCHING_METADATA),
                path=self.output_metadata_path
            )
        ]

    @property
    def executable(self) -> ModuleType:
        return nway_matching

    @property
    def dockerhub_repository_name(self):
        return 'ophys_nway_matching'

    @property
    def python_interpreter_path(self) -> str:
        return 'python'

    def _get_container_experiments_input(self):
        experiments = []
        experiment_ids = self.ophys_container.get_ophys_experiment_ids()
        for exp_id in experiment_ids:
            oe = OphysExperiment.from_id(id=exp_id)
            avg_projection_path = get_well_known_file_for_latest_run(
                ophys_experiment_id=exp_id,
                engine=engine,
                well_known_file_type=(
                    WellKnownFileTypeEnum.AVG_INTENSITY_PROJECTION_IMAGE),
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                workflow_step=WorkflowStepEnum.MOTION_CORRECTION
            )
            rois = [x.to_dict() for x in oe.rois]
            for roi in rois:
                # replace "mask" key with "mask_matrix"
                roi['mask_matrix'] = roi.pop('mask')

            experiments.append({
                'id': exp_id,
                'ophys_average_intensity_projection_image':
                    str(avg_projection_path),
                'cell_rois': rois
            })
        return experiments

    @staticmethod
    def save_matches_to_db(
            output_files: Dict[str, OutputFile],
            session: Session,
            run_id: int,
            **kwargs
    ):
        with open(output_files[
                    WellKnownFileTypeEnum.NWAY_CELL_MATCHING_METADATA.value
                  ].path) as f:
            matches = json.load(f)
        matches = matches['nway_matches']

        for i, matching_rois in enumerate(matches):
            for roi_id in matching_rois:
                match = NwayCellMatch(
                    nway_cell_matching_run_id=run_id,
                    ophys_roi_id=roi_id,
                    match_id=f'{run_id}_{i}'
                )
                session.add(match)
