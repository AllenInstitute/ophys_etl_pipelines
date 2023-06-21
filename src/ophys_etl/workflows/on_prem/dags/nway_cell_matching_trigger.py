import datetime

import logging
from airflow.decorators import task
from airflow.models.dag import dag
from airflow.operators.python import get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

from ophys_etl.workflows.db import engine
from sqlmodel import Session

from ophys_etl.workflows.utils.dag_utils import get_latest_dag_run
from ophys_etl.workflows.workflow_names import WorkflowNameEnum
from ophys_etl.workflows.workflow_step_runs import get_runs_completed_since, \
    get_completed
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

logger = logging.getLogger('airflow.task')


@dag(
    dag_id='nway_cell_matching_trigger',
    schedule='*/5 * * * *',  # every 5 minutes
    catchup=False,
    start_date=datetime.datetime.now()
)
def nway_cell_matching_trigger():
    """Checks for any ophys experiments that have completed segmentation
    since the last time this ran. If so, and
    1) all other experiments (with workflow_state of "passed" or "qc"
        from this container have also completed segmentation
        then we trigger nway cell matching for this container.
    """
    @task
    def trigger():
        last_run_datetime = get_latest_dag_run(
            dag_id='nway_cell_matching_trigger')
        if last_run_datetime is None:
            # this DAG hasn't been successfully run before
            # nothing to do
            return None
        with Session(engine) as session:
            segmentation_runs = get_runs_completed_since(
                session=session,
                since=last_run_datetime,
                workflow_name=WorkflowNameEnum.OPHYS_PROCESSING,
                workflow_step=WorkflowStepEnum.SEGMENTATION
            )
        ophys_experiment_ids = \
            [x.ophys_experiment_id for x in segmentation_runs]

        completed_ophys_containers = get_completed(
            ophys_experiment_ids=ophys_experiment_ids,
            workflow_step=WorkflowStepEnum.SEGMENTATION,
            level='ophys_container'
        )
        for ophys_container_id in completed_ophys_containers:
            logger.info(
                f'Triggering nway cell matching for ophys container '
                f'{ophys_container_id}')
            TriggerDagRunOperator(
                task_id='trigger_nway_cell_matching_for_ophys_container',
                trigger_dag_id='nway_cell_matching',
                conf={
                    'ophys_container_id': ophys_container_id
                }
            ).execute(context=get_current_context())

    trigger()


nway_cell_matching_trigger()

if __name__ == '__main__':
    nway_cell_matching_trigger().test()
