from airflow.sensors.base import PokeReturnValue
from ophys_etl.workflows.tasks import wait_for_dag_to_finish
from pathlib import Path

import datetime
from unittest import mock

import pytest
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from sqlmodel import Session

from ophys_etl.workflows.db.db_utils import save_job_run_to_db

from ophys_etl.workflows.ophys_experiment import OphysExperiment, \
    OphysContainer, Specimen, OphysSession

from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

from tests.workflows.conftest import MockSQLiteDB


class TestTasks(MockSQLiteDB):
    @mock.patch('ophys_etl.workflows.tasks.get_current_context')
    @mock.patch.object(OphysExperiment, 'from_id')
    @mock.patch('ophys_etl.workflows.tasks.get_latest_dag_run')
    @pytest.mark.parametrize(
        'dag_id, workflow_step, level',
        [
            pytest.param(
                # dag_id
                'decrosstalk',
                # workflow_step,
                WorkflowStepEnum.DECROSSTALK,
                # level
                'session'
            ),
            pytest.param(
                # dag_id
                'cell_classifier_inference',
                # workflow_step,
                WorkflowStepEnum.ROI_CLASSIFICATION_INFERENCE,
                # level
                'experiment'
            ),
            pytest.param(
                # dag_id
                'nway_cell_matching',
                # workflow_step,
                WorkflowStepEnum.NWAY_CELL_MATCHING,
                # level
                'container'
            )
        ]
    )
    def test_wait_for_dag_to_finish(
        self,
        mock_get_running_dag_run,
        mock_ophys_experiment,
        _,
        dag_id,
        workflow_step,
        level
    ):
        mock_get_running_dag_run.return_value = None
        mock_ophys_experiment.return_value = OphysExperiment(
            id=1,
            container=OphysContainer(id=1, specimen=Specimen(id='1')),
            session=OphysSession(id=1, specimen=Specimen(id='1')),
            equipment_name='MESO.1',
            full_genotype='',
            movie_frame_rate_hz=1,
            raw_movie_filename=Path(''),
            specimen=Specimen(id='1'),
            storage_directory=Path('')
        )

        # 1. First run we expect is_done to be False since it hasn't run yet
        with mock.patch('ophys_etl.workflows.tasks.engine', new=self._engine):
            res: PokeReturnValue = wait_for_dag_to_finish(
                    dag_id=dag_id,
                    workflow_step=workflow_step,
                    level=level
                ).function()
            assert not res.is_done

        # 2. Now we run the dag and we expect is_done to be True
        if level == 'experiment':
            kwargs = {
                'ophys_experiment_id':
                    mock_ophys_experiment.return_value.id
            }
        elif level == 'session':
            kwargs = {
                'ophys_session_id':
                    mock_ophys_experiment.return_value.session.id
            }
        elif level == 'container':
            kwargs = {
                'ophys_container_id':
                    mock_ophys_experiment.return_value.container.id
            }
        else:
            raise ValueError(f'invalid level {level}')

        with Session(self._engine) as session:
            save_job_run_to_db(
                start=datetime.datetime.now(),
                end=datetime.datetime.now(),
                module_outputs=[],
                sqlalchemy_session=session,
                storage_directory="foo",
                log_path="foo",
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                workflow_step_name=workflow_step,
                **kwargs
            )

        with mock.patch('ophys_etl.workflows.tasks.engine', new=self._engine):
            res: PokeReturnValue = wait_for_dag_to_finish(
                    dag_id=dag_id,
                    workflow_step=workflow_step,
                    level=level
                ).function()
            assert res.is_done
