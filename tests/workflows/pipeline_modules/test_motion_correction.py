import datetime
from pathlib import Path
from unittest.mock import patch, PropertyMock
from sqlmodel import Session, select

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.db.schemas import MotionCorrectionRun
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules.motion_correction import (
    MotionCorrectionModule,
)
from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysSession
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from tests.workflows.conftest import MockSQLiteDB


class TestMotionCorrectionModule(MockSQLiteDB):
    def test_save_metadata_to_db(self):
        _xy_offset_path = (
            Path(__file__).parent / "resources" / "rigid_motion_transform.csv"
        )
        with Session(self._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.MOTION_CORRECTION,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[
                    OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA
                        ),
                        path=_xy_offset_path,
                    )
                ],
                ophys_experiment_id="1",
                sqlalchemy_session=session,
                storage_directory="/foo",
                log_path="/foo",
                additional_steps=MotionCorrectionModule.save_metadata_to_db,
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
            )
        with Session(self._engine) as session:
            statement = select(MotionCorrectionRun)
            run = session.exec(statement).one()

        assert run.workflow_step_run_id == 1
        assert run.max_correction_up == 21.0
        assert run.max_correction_down == -20.0
        assert run.max_correction_left == 4.0
        assert run.max_correction_right == -3.0

    @patch.object(OphysExperiment, "motion_border", new_callable=PropertyMock)
    @patch.object(OphysExperiment, "rois", new_callable=PropertyMock)
    @patch.object(OphysSession, "output_dir", new_callable=PropertyMock)
    @patch.object(
        MotionCorrectionModule, "output_path", new_callable=PropertyMock
    )
    def test_inputs(
        self,
        mock_output_path,
        mock_output_dir,
        mock_oe_rois,
        mock_motion_border,
        temp_dir,
        mock_ophys_experiment,
        mock_motion_border_run,
        motion_corrected_ophys_movie_path,
        mock_rois,
    ):
        """Test that inputs are correctly formatted
        for input into the module."""
        mock_motion_border.return_value = mock_motion_border_run
        mock_oe_rois.return_value = mock_rois
        mock_output_path.return_value = temp_dir
        mock_output_dir.return_value = temp_dir

        mod = MotionCorrectionModule(
            docker_tag="main",
            ophys_experiment=mock_ophys_experiment,
            motion_corrected_ophys_movie_file=OutputFile(
                well_known_file_type=(
                    WellKnownFileTypeEnum.MOTION_CORRECTED_IMAGE_STACK
                ),
                path=motion_corrected_ophys_movie_path,
            ),
        )

        mod.inputs
