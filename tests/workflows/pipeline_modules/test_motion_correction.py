import datetime
from pathlib import Path

from conftest import MockSQLiteDB
from sqlmodel import Session, select

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.db.schemas import MotionCorrectionRun
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.pipeline_modules.motion_correction import (
    MotionCorrectionModule,
)
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum


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
