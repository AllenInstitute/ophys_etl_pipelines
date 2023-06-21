from collections import defaultdict
from types import ModuleType
from typing import List, Dict

import json

from sqlmodel import Session

from ophys_etl.modules.decrosstalk.decrosstalk_schema import \
    DecrosstalkInputSchema
from ophys_etl.workflows.db import engine
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum

from ophys_etl.workflows.workflow_step_runs import \
    get_well_known_file_for_latest_run

from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    ImagingPlaneGroup

from ophys_etl.modules import decrosstalk
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

DECROSSTALK_FLAGS = (
    'decrosstalk_invalid_raw',
    'decrosstalk_invalid_raw_active',
    'decrosstalk_invalid_unmixed',
    'decrosstalk_invalid_unmixed_active',
    'decrosstalk_ghost'
)


class DecrosstalkModule(PipelineModule):
    """Decrosstalk module"""

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.DECROSSTALK

    @property
    def inputs(self) -> Dict:
        ophys_experiments = [
            OphysExperiment.from_id(id=ophys_experiment_id) for
            ophys_experiment_id in self.ophys_session.ophys_experiment_ids
        ]
        ipg_ophys_experiment_map: Dict[
            ImagingPlaneGroup, List[OphysExperiment]] = \
            defaultdict(list)

        for ophys_experiment in ophys_experiments:
            ipg_ophys_experiment_map[
                ophys_experiment.imaging_plane_group] \
                .append(ophys_experiment)

        return DecrosstalkInputSchema().load(data={
            'ophys_session_id': self.ophys_session.id,
            'qc_output_dir': str(self.output_path),
            'coupled_planes': [{
                'ophys_imaging_plane_group_id': imaging_plane_group.id,
                'group_order': imaging_plane_group.group_order,
                'planes': [
                    self._get_plane_input(
                        ophys_experiment=ophys_experiment
                    )
                    for ophys_experiment in ipg_ophys_experiment_map[
                        imaging_plane_group]
                ]
            } for imaging_plane_group in ipg_ophys_experiment_map]
        })

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=WellKnownFileTypeEnum.DECROSSTALK_FLAGS,
                path=self.output_metadata_path
            )
        ]

    @property
    def executable(self) -> ModuleType:
        return decrosstalk

    @staticmethod
    def save_decrosstalk_flags_to_db(
            output_files: Dict[str, OutputFile],
            session: Session,
            run_id: int,
            **kwargs
    ):
        decrosstalk_file_path = output_files[
            WellKnownFileTypeEnum.DECROSSTALK_FLAGS.value
        ].path
        with open(decrosstalk_file_path) as f:
            output = json.load(f)

        roi_flags = defaultdict(list)
        coupled_planes = output['coupled_planes']

        for plane_group in coupled_planes:
            for plane in plane_group['planes']:
                ophys_experiment = OphysExperiment.from_id(
                    id=plane['ophys_experiment_id'])

                rois = ophys_experiment.rois

                # initializing all flags to False
                for roi in rois:
                    for flag in DECROSSTALK_FLAGS:
                        setattr(roi, f'is_{flag}', False)

                # accumulate flags for each roi
                for flag in DECROSSTALK_FLAGS:
                    for roi in plane[flag]:
                        roi_flags[roi].append(flag)

                # Update flags for each roi that has been flagged
                for roi in rois:
                    flags = roi_flags.get(roi.id, [])
                    for flag in flags:
                        setattr(roi, f'is_{flag}', True)

                # Update flags in the database
                for roi in rois:
                    session.add(roi)

    def _get_plane_input(
            self,
            ophys_experiment: OphysExperiment
    ):
        """Gets the inputs for a single plane"""
        return {
            'ophys_experiment_id': ophys_experiment.id,
            'motion_corrected_stack': str(
                get_well_known_file_for_latest_run(
                    ophys_experiment_id=ophys_experiment.id,
                    engine=engine,
                    well_known_file_type=(
                        WellKnownFileTypeEnum.
                        MOTION_CORRECTED_IMAGE_STACK),
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    workflow_step=WorkflowStepEnum.MOTION_CORRECTION
                )
            ),
            'maximum_projection_image_file': str(
                get_well_known_file_for_latest_run(
                    ophys_experiment_id=ophys_experiment.id,
                    engine=engine,
                    well_known_file_type=(
                        WellKnownFileTypeEnum.
                        MAX_INTENSITY_PROJECTION_IMAGE),
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    workflow_step=WorkflowStepEnum.MOTION_CORRECTION
                )
            ),
            'output_roi_trace_file': str(
                    self.output_path /
                    f'ophys_experiment_{ophys_experiment.id}_roi_traces.h5'),
            'output_neuropil_trace_file': str(
                    self.output_path /
                    f'ophys_experiment_{ophys_experiment.id}_'
                    f'neuropil_traces.h5'),
            'motion_border': ophys_experiment.motion_border.to_dict(),
            'rois': [x.to_dict() for x in ophys_experiment.rois]
        }
