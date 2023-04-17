"""Motion correction pipeline module"""
from types import ModuleType
from typing import List, Dict

from ophys_etl.modules import suite2p_registration
from ophys_etl.workflows.workflow_steps import WorkflowStep
from sqlmodel import Session

from ophys_etl.utils.motion_border import get_max_correction_from_file
from ophys_etl.workflows.db.schemas import MotionCorrectionRun
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.well_known_file_types import \
    WellKnownFileType


class MotionCorrectionModule(PipelineModule):
    """Wrapper around motion correction module"""

    @property
    def _executable(self) -> ModuleType:
        return suite2p_registration

    @property
    def queue_name(self) -> WorkflowStep:
        return WorkflowStep.MOTION_CORRECTION

    @property
    def inputs(self):
        module_args = {
            'movie_frame_rate_hz': self._ophys_experiment.movie_frame_rate_hz,
            'suite2p_args': {
                'h5py': str(self._ophys_experiment.storage_directory /
                            self._ophys_experiment.raw_movie_filename)
            },
            'motion_corrected_output': (
                str(self.output_path /
                    f'{self._ophys_experiment.id}_suite2p_motion_output.h5')),
            'motion_diagnostics_output': (
                str(self.output_path /
                    f'{self._ophys_experiment.id}_'
                    f'suite2p_rigid_motion_transform.csv')),
            'max_projection_output': (
                str(self.output_path / f'{self._ophys_experiment.id}_'
                                       f'suite2p_maximum_projection.png')),
            'avg_projection_output': (
                str(self.output_path / f'{self._ophys_experiment.id}_'
                                       f'suite2p_average_projection.png')),
            'registration_summary_output': (
                str(self.output_path / f'{self._ophys_experiment.id}_'
                                       f'suite2p_registration_summary.png')),
            'motion_correction_preview_output': (
                str(self.output_path / f'{self._ophys_experiment.id}_'
                                       f'suite2p_motion_preview.webm'))
        }
        return module_args

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileType.MOTION_CORRECTED_IMAGE_STACK),
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_suite2p_motion_output.h5')
            ),
            OutputFile(
                well_known_file_type=(
                    WellKnownFileType.MAX_INTENSITY_PROJECTION_IMAGE),
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_'
                      f'suite2p_maximum_projection.png')
            ),
            OutputFile(
                well_known_file_type=(
                    WellKnownFileType.MOTION_X_Y_OFFSET_DATA),
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_'
                      f'suite2p_rigid_motion_transform.csv')
            ),
            OutputFile(
                well_known_file_type=(
                    WellKnownFileType.AVG_INTENSITY_PROJECTION_IMAGE),
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_'
                      f'suite2p_average_projection.png')
            ),
            OutputFile(
                well_known_file_type=(
                    WellKnownFileType.REGISTRATION_SUMMARY_IMAGE),
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_'
                      f'suite2p_registration_summary.png')
            ),
            OutputFile(
                well_known_file_type=(
                    WellKnownFileType.MOTION_PREVIEW),
                path=(self.output_path /
                      f'{self._ophys_experiment.id}_'
                      f'suite2p_motion_preview.webm')
            ),
        ]

    @staticmethod
    def save_metadata_to_db(
            output_files: Dict[str, OutputFile],
            session: Session,
            run_id: int
    ):
        """
        Saves motion correction run results to db

        Parameters
        ----------
        output_files
            Files output by this module
        session
            sqlalchemy session
        run_id
            workflow step run id
        """
        offset_file_path = \
            output_files[WellKnownFileType.MOTION_X_Y_OFFSET_DATA.value].path
        maximum_motion_shift = get_max_correction_from_file(
            input_csv=offset_file_path
        )
        run = MotionCorrectionRun(
            workflow_step_run_id=run_id,
            max_correction_up=maximum_motion_shift.up,
            max_correction_down=maximum_motion_shift.down,
            max_correction_left=maximum_motion_shift.left,
            max_correction_right=maximum_motion_shift.right
        )
        session.add(run)
