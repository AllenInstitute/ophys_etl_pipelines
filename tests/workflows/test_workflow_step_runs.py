import datetime
from pathlib import Path

import pytest

from ophys_etl.test_utils.workflow_utils import setup_app_config

setup_app_config(
    ophys_workflow_app_config_path=(
            Path(__file__).parent / 'resources' / 'config.yml'),
    test_di_base_model_path=Path(__file__).parent / 'resources' / 'di_model.h5'
)

from ophys_etl.workflows.well_known_file_types import WellKnownFileType # noqa E402

from ophys_etl.workflows.output_file import OutputFile  # noqa E402
from ophys_etl.workflows.workflow_step_runs import get_latest_run, \
    get_well_known_file_for_latest_run  # noqa E402
from ophys_etl.workflows.workflow_steps import WorkflowStep # noqa E402

from ophys_etl.workflows.db.db_utils import save_job_run_to_db  # noqa E402
from sqlmodel import Session    # noqa E402

from ophys_etl.workflows.workflow_names import WorkflowName # noqa E402
from ophys_etl.test_utils.db_base import MockSQLiteDB

class TestWorkflowStepRuns(MockSQLiteDB):

    @pytest.mark.parametrize('ophys_experiment_id', (None, '1'))
    def test__get_latest_run(self, ophys_experiment_id):
        """Test that the latest run is the most recently added"""
        workflow_name = WorkflowName.OPHYS_PROCESSING
        workflow_step_name = WorkflowStep.SEGMENTATION

        with Session(self._engine) as session:
            for i in range(1, 3):
                save_job_run_to_db(
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[],
                    sqlalchemy_session=session,
                    ophys_experiment_id=ophys_experiment_id,
                    storage_directory='foo',
                    log_path='foo',
                    workflow_name=workflow_name,
                    workflow_step_name=workflow_step_name
                )
                latest_run = get_latest_run(
                    session=session,
                    workflow_name=workflow_name,
                    workflow_step=workflow_step_name
                )
                assert latest_run == i

    @pytest.mark.parametrize('ophys_experiment_id', (None, '1'))
    def test_get_well_known_file_for_latest_run(self, ophys_experiment_id):
        workflow_name = WorkflowName.OPHYS_PROCESSING
        workflow_step_name = WorkflowStep.SEGMENTATION
        well_known_file_type = WellKnownFileType.OPHYS_ROIS

        paths = []
        with Session(self._engine) as session:
            for i in range(2):
                output_file_path = Path(self._tmp_dir) / f'{i}.txt'
                paths.append(output_file_path)
                with open(output_file_path, 'w') as f:
                    f.write('')

                save_job_run_to_db(
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[
                        OutputFile(
                            well_known_file_type=well_known_file_type,
                            path=output_file_path
                        )
                    ],
                    sqlalchemy_session=session,
                    ophys_experiment_id=ophys_experiment_id,
                    storage_directory='foo',
                    log_path='foo',
                    workflow_name=workflow_name,
                    workflow_step_name=workflow_step_name
                )

        path = get_well_known_file_for_latest_run(
            engine=self._engine,
            well_known_file_type=well_known_file_type,
            workflow_step=workflow_step_name,
            workflow_name=workflow_name,
            ophys_experiment_id=ophys_experiment_id
        )
        assert path == paths[-1]
