import datetime
from pathlib import Path

import pytest
from sqlmodel import Session

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from ophys_etl.workflows.output_file import OutputFile
from ophys_etl.workflows.well_known_file_types import WellKnownFileTypeEnum
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import (
    get_latest_workflow_step_run,
    get_well_known_file_for_latest_run,
)
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from tests.workflows.conftest import MockSQLiteDB


class TestWorkflowStepRuns(MockSQLiteDB):
    @pytest.mark.parametrize("ophys_experiment_id", (None, "1"))
    def test__get_latest_run(self, ophys_experiment_id):
        """Test that the latest run is the most recently added"""

        with Session(self._engine) as session:
            for segmentation_run_id in range(1, 7, 2):
                save_job_run_to_db(
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[],
                    sqlalchemy_session=session,
                    ophys_experiment_id=ophys_experiment_id,
                    storage_directory="foo",
                    log_path="foo",
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    workflow_step_name=WorkflowStepEnum.SEGMENTATION,
                )
                save_job_run_to_db(
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[],
                    sqlalchemy_session=session,
                    ophys_experiment_id=ophys_experiment_id,
                    storage_directory="foo",
                    log_path="foo",
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    workflow_step_name=WorkflowStepEnum.MOTION_CORRECTION,
                )
                latest_run = get_latest_workflow_step_run(
                    session=session,
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    workflow_step=WorkflowStepEnum.SEGMENTATION,
                )
                assert latest_run == segmentation_run_id

    @pytest.mark.parametrize("ophys_experiment_id", (None, "1"))
    def test_get_well_known_file_for_latest_run(self, ophys_experiment_id):
        workflow_name = WorkflowNameEnum.OPHYS_PROCESSING
        workflow_step_name = WorkflowStepEnum.SEGMENTATION
        well_known_file_type = WellKnownFileTypeEnum.OPHYS_ROIS

        paths = []
        with Session(self._engine) as session:
            for i in range(2):
                output_file_path = Path(self._tmp_dir) / f"{i}.txt"
                paths.append(output_file_path)
                with open(output_file_path, "w") as f:
                    f.write("")

                save_job_run_to_db(
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[
                        OutputFile(
                            well_known_file_type=well_known_file_type,
                            path=output_file_path,
                        )
                    ],
                    sqlalchemy_session=session,
                    ophys_experiment_id=ophys_experiment_id,
                    storage_directory="foo",
                    log_path="foo",
                    workflow_name=workflow_name,
                    workflow_step_name=workflow_step_name,
                )

        path = get_well_known_file_for_latest_run(
            engine=self._engine,
            well_known_file_type=well_known_file_type,
            workflow_step=workflow_step_name,
            workflow_name=workflow_name,
            ophys_experiment_id=ophys_experiment_id,
        )
        assert path == paths[-1]
