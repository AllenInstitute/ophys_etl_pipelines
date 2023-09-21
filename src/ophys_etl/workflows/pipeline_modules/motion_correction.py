"""Motion correction pipeline module"""
from pathlib import Path
from types import ModuleType
from typing import Dict, List

import h5py

from ophys_etl.workflows.app_config.app_config import app_config
from sqlmodel import Session

from ophys_etl.modules import suite2p_registration
from ophys_etl.modules.suite2p_registration.schemas import Suite2PRegistrationInputSchema  # noqa: E501
from ophys_etl.utils.motion_border import get_max_correction_from_file
from ophys_etl.workflows.db.schemas import MotionCorrectionRun
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_module import PipelineModule
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


class MotionCorrectionModule(PipelineModule):
    """Wrapper around motion correction module"""

    def __init__(self,
                 ophys_experiment,
                 prevent_file_overwrites: bool = True,
                 **kwargs):
        super().__init__(
            ophys_experiment=ophys_experiment,
            prevent_file_overwrites=prevent_file_overwrites,
            **kwargs,
        )

    @property
    def executable(self) -> ModuleType:
        return suite2p_registration

    @property
    def queue_name(self) -> WorkflowStepEnum:
        return WorkflowStepEnum.MOTION_CORRECTION

    @property
    def module_schema(self) -> Suite2PRegistrationInputSchema:
        return Suite2PRegistrationInputSchema()

    @property
    def inputs(self) -> Dict:
        if app_config.is_debug:
            movie_file_path = self._construct_short_movie()
        else:
            movie_file_path = (
                    self._ophys_experiment.storage_directory /
                    self._ophys_experiment.raw_movie_filename)
        return {
            "movie_frame_rate_hz": self._ophys_experiment.movie_frame_rate_hz,
            "suite2p_args": {
                "h5py": str(movie_file_path),
                "nonrigid": (
                    app_config.pipeline_steps.motion_correction.nonrigid)
            },
            "motion_corrected_output": (
                str(
                    self.output_path
                    / f"{self._ophys_experiment.id}_suite2p_motion_output.h5"
                )
            ),
            "motion_diagnostics_output": (
                str(
                    self.output_path / f"{self._ophys_experiment.id}_"
                    f"suite2p_rigid_motion_transform.csv"
                )
            ),
            "max_projection_output": (
                str(
                    self.output_path / f"{self._ophys_experiment.id}_"
                    f"suite2p_maximum_projection.png"
                )
            ),
            "avg_projection_output": (
                str(
                    self.output_path / f"{self._ophys_experiment.id}_"
                    f"suite2p_average_projection.png"
                )
            ),
            "registration_summary_output": (
                str(
                    self.output_path / f"{self._ophys_experiment.id}_"
                    f"suite2p_registration_summary.png"
                )
            ),
            "motion_correction_preview_output": (
                str(
                    self.output_path / f"{self._ophys_experiment.id}_"
                    f"suite2p_motion_preview.webm"
                )
            ),
        }

    @property
    def outputs(self) -> List[OutputFile]:
        return [
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                ),
                path=(
                    self.output_path
                    / f"{self._ophys_experiment.id}_suite2p_motion_output.h5"
                ),
            ),
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.MAX_INTENSITY_PROJECTION_IMAGE
                ),
                path=(
                    self.output_path / f"{self._ophys_experiment.id}_"
                    f"suite2p_maximum_projection.png"
                ),
            ),
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA
                ),
                path=(
                    self.output_path / f"{self._ophys_experiment.id}_"
                    f"suite2p_rigid_motion_transform.csv"
                ),
            ),
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.AVG_INTENSITY_PROJECTION_IMAGE
                ),
                path=(
                    self.output_path / f"{self._ophys_experiment.id}_"
                    f"suite2p_average_projection.png"
                ),
            ),
            OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.REGISTRATION_SUMMARY_IMAGE
                ),
                path=(
                    self.output_path / f"{self._ophys_experiment.id}_"
                    f"suite2p_registration_summary.png"
                ),
            ),
            OutputFile(
                well_known_file_type=(WellKnownFileTypeEnum.MOTION_PREVIEW),
                path=(
                    self.output_path / f"{self._ophys_experiment.id}_"
                    f"suite2p_motion_preview.webm"
                ),
            ),
        ]

    @staticmethod
    def save_metadata_to_db(
        output_files: Dict[str, OutputFile],
        session: Session,
        run_id: int,
        **kwargs
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
        offset_file_path = output_files[
            WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA.value
        ].path
        maximum_motion_shift = get_max_correction_from_file(
            input_csv=offset_file_path
        )
        run = MotionCorrectionRun(
            workflow_step_run_id=run_id,
            max_correction_up=maximum_motion_shift.up,
            max_correction_down=maximum_motion_shift.down,
            max_correction_left=maximum_motion_shift.left,
            max_correction_right=maximum_motion_shift.right,
        )
        session.add(run)

    def _construct_short_movie(self) -> Path:
        """In debug mode, we construct a short movie to work on and write
        it to app_config.output_dir

        Returns
        -------
        Path
            Path to short movie
        """
        movie_file_path = (
                self._ophys_experiment.storage_directory /
                self._ophys_experiment.raw_movie_filename)
        with h5py.File(movie_file_path, 'r') as f:
            mov = f['data'][:200]

        out_path = app_config.output_dir / \
            f'{self.ophys_experiment.id}_debug_movie.h5'
        with h5py.File(out_path, 'w') as f:
            f.create_dataset(name='data', data=mov)
        return out_path
