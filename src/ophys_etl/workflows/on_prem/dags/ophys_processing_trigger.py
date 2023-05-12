import datetime
import logging
from typing import List

from airflow.decorators import task
from airflow.models.dag import dag
from airflow.operators.python import get_current_context
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from ophys_etl.workflows.app_config.app_config import app_config
from ophys_etl.workflows.utils.dag_utils import get_most_recent_run

from ophys_etl.workflows.utils.lims_utils import LIMSDB


def _get_all_ophys_experiments_completed_since(
        since: datetime.datetime
) -> List[int]:
    """
    Get all ophys experiments which successfully completed one of the
    `LIMS_OPHYS_PROCESSING_TRIGGER_QUEUES` since `since`

    Parameters
    ----------
    since
        Check for jobs that completed since this time

    Returns
    -------
    List of ophys experiment LIMS ids

    """
    trigger_queues = app_config.ophys_processing_trigger.lims_trigger_queues
    if len(trigger_queues) == 1:
        trigger_queues = trigger_queues[0]
        queue_query = f'jq.name = \'{trigger_queues}\''
    else:
        queue_query = f'jq.name in {trigger_queues}'

    query = f'''
        SELECT oe.id
        FROM jobs
        JOIN job_queues jq on jq.id = jobs.job_queue_id
        JOIN ophys_sessions os on os.id = jobs.enqueued_object_id
        JOIN ophys_experiments oe on oe.ophys_session_id = os.id
        JOIN job_states js on js.id = jobs.job_state_id
        WHERE {queue_query} AND
                js.name = 'SUCCESS' AND
                completed_at >= '{since}'
    '''
    lims_db = LIMSDB()
    res = lims_db.query(query=query)
    ophys_experiment_ids = [x['id'] for x in res]
    return ophys_experiment_ids


logger = logging.getLogger('airflow.task')


@dag(
    dag_id='ophys_processing_trigger',
    schedule='*/5 * * * *',     # every 5 minutes
    catchup=False,
    start_date=datetime.datetime.now()
)
def ophys_processing_trigger():

    @task
    def trigger():
        last_run_datetime = get_most_recent_run(
            dag_id='ophys_processing_trigger')
        if last_run_datetime is None:
            # this DAG hasn't been successfully run before
            # nothing to do
            return None
        ophys_experiment_ids = _get_all_ophys_experiments_completed_since(
            since=last_run_datetime
        )
        for ophys_experiment_id in ophys_experiment_ids:
            logger.info(f'Triggering ophys_processing DAG for '
                        f'{ophys_experiment_id}')
            TriggerDagRunOperator(
                task_id='trigger_ophys_processing_for_ophys_experiment',
                trigger_dag_id='ophys_processing',
                conf={
                    'ophys_experiment_id': ophys_experiment_id
                }
            ).execute(context=get_current_context())
    trigger()


ophys_processing_trigger()
