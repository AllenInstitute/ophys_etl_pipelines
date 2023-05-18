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
from ophys_etl.workflows.workflow_step_runs import get_runs_completed_since
from ophys_etl.workflows.workflow_steps import WorkflowStepEnum

logger = logging.getLogger('airflow.task')


@dag(
    dag_id='cell_classifier_inference_trigger',
    schedule='*/5 * * * *',  # every 5 minutes
    catchup=False,
    start_date=datetime.datetime.now()
)
def cell_classifier_inference_trigger():
    """Triggers cell classifier inference when an ophys experiment has
    completed segmentation"""

    @task
    def trigger():
        last_run_datetime = get_latest_dag_run(
            dag_id='cell_classifier_inference_trigger')
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

        for ophys_experiment_id in ophys_experiment_ids:
            logger.info(
                f'Triggering cell classifier inference for ophys experiment '
                f'{ophys_experiment_id}')
            TriggerDagRunOperator(
                task_id='trigger_cell_classifier_inference_for_ophys_'
                        'experiment',
                trigger_dag_id='cell_classifier_inference',
                conf={
                    'ophys_experiment_id': ophys_experiment_id
                }
            ).execute(context=get_current_context())

    trigger()


cell_classifier_inference_trigger()
