import datetime
import os
import shutil
from pathlib import Path

import tempfile

from sqlmodel import select, Session

from ophys_etl.test_utils.workflow_utils import setup_app_config
from ophys_etl.workflows.workflow_names import WorkflowNameEnum

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent.parent / 'resources' / 'config.yml'),
    test_di_base_model_path=Path(__file__).parent.parent / 'resources' /
    'di_model.h5'
)

from ophys_etl.workflows.pipeline_modules.motion_correction import \
    MotionCorrectionModule # noqa E402
from ophys_etl.workflows.db.db_utils import save_job_run_to_db # noqa E402
from ophys_etl.workflows.db.initialize_db import InitializeDBRunner # noqa E402
from ophys_etl.workflows.db.schemas import MotionCorrectionRun # noqa E402
from ophys_etl.workflows.output_file import OutputFile  # noqa E402
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum # noqa E402
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from conftest import MockSQLiteDB

class TestMotionCorrectionModule(MockSQLiteDB):

    def test_save_metadata_to_db(self):
        _xy_offset_path = \
            Path(__file__).parent / 'resources' / 'rigid_motion_transform.csv'
        with Session(self._engine) as session:
            save_job_run_to_db(
                workflow_step_name=WorkflowStepEnum.MOTION_CORRECTION,
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[OutputFile(
                        well_known_file_type=(
                            WellKnownFileTypeEnum.MOTION_X_Y_OFFSET_DATA),
                        path=_xy_offset_path
                    )
                ],
                ophys_experiment_id='1',
                sqlalchemy_session=session,
                storage_directory='/foo',
                log_path='/foo',
                additional_steps=MotionCorrectionModule.save_metadata_to_db,
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING
            )
        with Session(self._engine) as session:
            statement = select(MotionCorrectionRun)
            run = session.exec(statement).one()

        assert run.workflow_step_run_id == 1
        assert run.max_correction_up == 21.0
        assert run.max_correction_down == -20.0
        assert run.max_correction_left == 4.0
        assert run.max_correction_right == -3.0
