import datetime
from unittest.mock import patch

from ophys_etl.workflows.workflow_step_runs import \
    get_completed
from ophys_etl.workflows.workflow_names import WorkflowNameEnum

from ophys_etl.workflows.db.db_utils import save_job_run_to_db
from sqlmodel import Session

from ophys_etl.workflows.workflow_steps import WorkflowStepEnum
from tests.workflows.conftest import MockSQLiteDB


class TestDecrosstalkTrigger(MockSQLiteDB):
    def setup(self):
        super().setup()

        with Session(self._engine) as session:
            mock_completed_segmentation = ['a', 'b', 'c']
            for oe_id in mock_completed_segmentation:
                save_job_run_to_db(
                    start=datetime.datetime.now(),
                    end=datetime.datetime.now(),
                    module_outputs=[],
                    sqlalchemy_session=session,
                    ophys_experiment_id=oe_id,
                    storage_directory="foo",
                    log_path="foo",
                    workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                    workflow_step_name=WorkflowStepEnum.SEGMENTATION,
                )

    @patch('ophys_etl.workflows.workflow_step_runs.'
           'get_session_experiment_id_map')
    def test__get_completed_ophys_sessions(
            self, mock_session_exp_map):
        mock_session_exp_map.return_value = [
            {'ophys_session_id': 1, 'ophys_experiment_id': 'a'},
            {'ophys_session_id': 1, 'ophys_experiment_id': 'b'},
            {'ophys_session_id': 2, 'ophys_experiment_id': 'c'},
            {'ophys_session_id': 2, 'ophys_experiment_id': 'd'},
        ]

        with patch('ophys_etl.workflows.workflow_step_runs.engine',
                   new=self._engine):
            completed_sessions = get_completed(
                ophys_experiment_ids=['a', 'c'],
                workflow_step=WorkflowStepEnum.SEGMENTATION
            )
        assert completed_sessions == [1]

    @patch('ophys_etl.workflows.workflow_step_runs.'
           'get_session_experiment_id_map')
    def test__get_completed_ophys_sessions_none_complete(
            self, mock_session_exp_map):
        mock_session_exp_map.return_value = [
            {'ophys_session_id': 2, 'ophys_experiment_id': 'c'},
            {'ophys_session_id': 2, 'ophys_experiment_id': 'd'},
        ]

        with patch('ophys_etl.workflows.workflow_step_runs.engine',
                   new=self._engine):
            completed_sessions = get_completed(
                ophys_experiment_ids=['c'],
                workflow_step=WorkflowStepEnum.SEGMENTATION
            )
        assert completed_sessions == []

    @patch('ophys_etl.workflows.workflow_step_runs.'
           'get_session_experiment_id_map')
    def test__get_completed_ophys_sessions_all_complete(
            self, mock_session_exp_map):
        mock_session_exp_map.return_value = [
            {'ophys_session_id': 1, 'ophys_experiment_id': 'a'},
            {'ophys_session_id': 1, 'ophys_experiment_id': 'b'},
            {'ophys_session_id': 2, 'ophys_experiment_id': 'c'},
        ]

        with patch('ophys_etl.workflows.workflow_step_runs.engine',
                   new=self._engine):
            completed_sessions = get_completed(
                ophys_experiment_ids=['a', 'c'],
                workflow_step=WorkflowStepEnum.SEGMENTATION
            )
        assert completed_sessions == [1, 2]
